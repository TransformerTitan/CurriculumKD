import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

# Visual complexity (dummy): mean gradient magnitude approximation
def compute_visual_complexity(images):
    # images shape: (batch, channels, height, width), values [0,1]
    grad_x = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean(dim=[1,2,3])
    grad_y = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean(dim=[1,2,3])
    complexity = (grad_x + grad_y).cpu()
    return complexity

# Curriculum Scheduler as before
class CurriculumScheduler:
    def __init__(self, teacher_names, max_epochs):
        self.teacher_names = teacher_names
        self.max_epochs = max_epochs

    def select_teachers(self, epoch, complexity_scores):
        batch_size = complexity_scores.size(0)
        num_teachers = len(self.teacher_names)
        weights = torch.zeros(batch_size, num_teachers)

        if epoch < self.max_epochs * 0.33:
            active_teachers = [0, 1]
        elif epoch < self.max_epochs * 0.66:
            active_teachers = [1, 2]
        else:
            active_teachers = [2, 3]

        for i in range(batch_size):
            c = complexity_scores[i].item()
            for t in active_teachers:
                if t == 0:
                    weights[i, t] = max(0.5 - 0.5 * c, 0)
                elif t == 1:
                    weights[i, t] = 0.3
                elif t == 2:
                    weights[i, t] = min(0.5 * c + 0.2, 1.0)
                elif t == 3:
                    weights[i, t] = 0.2
            w_sum = weights[i, active_teachers].sum()
            if w_sum > 0:
                weights[i, active_teachers] /= w_sum

        return active_teachers, weights.to(complexity_scores.device)

# Distillation loss (KL div)
def distillation_loss(student_logits, teacher_logits_list, weights, temperature=4.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    total_loss = 0.0

    for i, teacher_logits in enumerate(teacher_logits_list):
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1)
        total_loss += (weights[:, i] * kl).mean()

    return total_loss * (temperature ** 2)

# Load HF pretrained ViT models as teachers and student
def get_teacher_models(device):
    teacher_names = [
        "google/vit-base-patch16-224-in21k",  # edge expert (pretend)
        "google/vit-large-patch16-224-in21k", # texture expert
        "google/vit-base-patch32-384",        # semantic expert
        "facebook/deit-base-patch16-224"      # multi-scale expert
    ]
    teachers = []
    for name in teacher_names:
        model = ViTForImageClassification.from_pretrained(name).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
    return teachers

def get_student_model(device):
    model = ViTForImageClassification.from_pretrained("google/vit-tiny-patch16-224").to(device)
    return model

# Prepare HF datasets with transforms (CIFAR-100 for example)
def preprocess(example, feature_extractor):
    image = example["img"]
    inputs = feature_extractor(images=image, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"].squeeze()}

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def train(model, teachers, scheduler, dataloader, optimizer, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Compute visual complexity (on [0,1] images)
            complexity_scores = compute_visual_complexity(images)

            # Scheduler selects teachers and weights
            active_teachers, weights = scheduler.select_teachers(epoch, complexity_scores)

            # Student logits
            student_outputs = model(images)
            student_logits = student_outputs.logits

            # Teacher logits
            teacher_logits_list = []
            for idx in active_teachers:
                with torch.no_grad():
                    teacher_logits = teachers[idx](images).logits
                teacher_logits_list.append(teacher_logits)

            # Compute losses
            distill_loss = distillation_loss(student_logits, teacher_logits_list, weights)
            cls_loss = criterion(student_logits, labels)

            alpha = 0.7
            loss = alpha * distill_loss + (1 - alpha) * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    import datasets
    from transformers import ViTFeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and feature extractor
    dataset = datasets.load_dataset("cifar100")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-tiny-patch16-224")

    dataset = dataset["train"].map(lambda x: preprocess(x, feature_extractor), remove_columns=["img", "coarse_label", "fine_label"])
    dataset.set_format(type="torch", columns=["pixel_values", "label"])

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Models and scheduler
    teachers = get_teacher_models(device)
    student = get_student_model(device)
    max_epochs = 10
    teacher_names = ['edge', 'texture', 'semantic', 'multi_scale']
    scheduler = CurriculumScheduler(teacher_names, max_epochs)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)

    # Train
    train(student, teachers, scheduler, train_loader, optimizer, device, epochs=max_epochs)
