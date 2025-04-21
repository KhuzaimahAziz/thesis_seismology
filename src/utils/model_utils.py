import logging
import torch
import seisbench.models as sbm
log = logging.getLogger(__name__)

def setup_model(cfg):
    model_name = cfg.model.name

    if not hasattr(sbm, model_name):
        log.error(f"❌ Model '{model_name}' is not found in seisbench.models.")
        log.info(f"🧠 Available models are: {', '.join(m for m in dir(sbm) if not m.startswith('_'))}")
        raise ValueError(f"Model '{model_name}' not found in seisbench.models")

    model_class = getattr(sbm, model_name)
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}

    log.info(f"🔧 Instantiating model: {model_name} with args: {model_kwargs}")
    model = model_class(**model_kwargs)

    if torch.cuda.is_available():
        model.cuda()
        log.info("🚀 Running on GPU")
    else:
        log.info("💻 Running on CPU")

    return model

def setup_optimizer(model, optimizer_cfg):
    optimizer_class = getattr(torch.optim, optimizer_cfg.name)
    return optimizer_class(model.parameters(), **optimizer_cfg.params)

def loss_fn(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    log.info(f"🔁 Training on {size} samples")
    for batch_id, batch in enumerate(dataloader):
        pred = model(batch["X"].float().to(model.device))
        loss = loss_fn(pred, batch["y"].float().to(model.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            log.info(f"[Batch {batch_id}] Loss: {loss.item():.6f}")

def test_loop(model, dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()
    avg_loss = test_loss / num_batches
    log.info(f"Test avg loss: {avg_loss:>8f}\n")
    return avg_loss
