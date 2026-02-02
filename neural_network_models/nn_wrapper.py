import torch

class NNWrapper():
    def __init__(self, model, loss_func, loss_optimization_func, epoch_limit, delta, patience_limit):
        self.model = model

        # loss calculation config
        self.loss_func = loss_func
        self.loss_optimization_func = loss_optimization_func

        # training stop condition config
        self.epoch_limit = epoch_limit
        self.delta = delta
        self.patience_limit = patience_limit


    def model_training(self):
        training_losses = []
        validation_losses = []

        best_validation_loss = float("inf")
        best_model_state = None

        patience_tries = 0

        # training process
        for epoch in range(self.epoch_limit):
            avg_training_loss = self.training_step()
            training_losses.append(avg_training_loss)

            avg_validation_loss = self.validation_step()
            validation_losses.append(avg_validation_loss)
            
            # printing losses
            if (epoch + 1) % 10 == 0:
                print(f"[{epoch + 1}] Training loss: {avg_training_loss:.6f} | Validation loss: {avg_validation_loss:.6f}")

            # stop condition
            if best_model_state is None or avg_validation_loss < best_validation_loss - self.delta:
                best_validation_loss = avg_validation_loss
                best_model_state = self.model.state_dict()
                patience_tries = 0
            else:
                patience_tries += 1

                if patience_tries >= self.patience_limit:
                    print(f"Early stopping at epoch {epoch + 1} as there was no improvement for {self.patience_limit} epochs")
                    break
        
        return best_model_state, best_validation_loss, training_losses, validation_losses


    def training_step(self, train_loader):
        epoch_training_loss = 0.0

        self.model.train()
        for batch_vectors, batch_scalars in train_loader:
            # pass forward
            predictions = self.model(batch_vectors)
            loss = self.loss_func(predictions, batch_scalars)
            
            # back-propagation
            self.loss_optimization_func.zero_grad()
            loss.backward()
            self.loss_optimization_func.step()
            
            epoch_training_loss += loss.item() * batch_vectors.size(0)

        # average batch loss
        epoch_avg_loss /= len(train_loader.dataset)
        return epoch_avg_loss


    def validation_step(self, loss_func, val_loader):
        epoch_validation_loss = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for batch_vectors, batch_scalars in val_loader:
                predictions = self.model(batch_vectors)

                loss = loss_func(predictions, batch_scalars)
                epoch_validation_loss += loss.item() * batch_vectors.size(0)

        epoch_avg_loss /= len(val_loader.dataset)
        return epoch_avg_loss