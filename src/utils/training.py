import torch
import torch.nn as nn
import torch.optim as optim
def train_cnn(model, train_loader, val_loader, device, learning_rate=0.0001, epochs=5):
    """Train the CNN model."""

    model.train()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            # Move to GPU if available
            data = data.to(device)
            label = label.to(device)

            # Forward pass
            output = model(data)[0]
            loss = criterion(output, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += accuracy / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(
            f"Epoch : {epoch+1}, Training Accuracy : {epoch_accuracy}, Training Loss : {epoch_loss}"
        )

        # Validation
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)[0]
                val_loss = criterion(val_output, label)

                accuracy = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += accuracy / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print(
                f"Epoch : {epoch+1}, Validation Accuracy : {epoch_val_accuracy}, Validation Loss : {epoch_val_loss}"
            )

    return model

