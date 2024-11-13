import torch
import copy

class Client:
    def __init__(self, client_id, train_loader, test_loader, model, criterion, optimizer, device, byzantine):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.byzantine = byzantine

    def train(self, epochs=1):
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if self.byzantine == 'label_flipping':
                    # Shuffle the labels to simulate a label-flipping attack
                    labels = labels[torch.randperm(labels.size(0))]
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.byzantine == 'sign_flipping':
                    # Flip the sign of the gradients
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = -param.grad

                self.optimizer.step()

    def evaluate(self):
        self.model.to(self.device)  
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def get_model(self):
        return copy.deepcopy(self.model)

    def set_model(self, new_model):
        self.model.load_state_dict(copy.deepcopy(new_model).state_dict())
        

    def get_ID(self):
        return self.client_id