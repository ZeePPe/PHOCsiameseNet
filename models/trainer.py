import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda_id, log_interval, metrics=[],
        start_epoch=0, save_model="model.pth", out_features=648, save_each_epoch=False, max_batchsize=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    best_score = -1
    best_val_los = -1
    best_train_los = -1
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        #scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, out_features, loss_fn, optimizer, cuda_id, log_interval, metrics, max_batchsize=max_batchsize)

        message = '     - Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, out_features, loss_fn, cuda_id, metrics)

        norm_val_loss = val_loss/len(val_loader)

        message += '\n     - Epoch:{}/{}. Validation set: Average loss:{:.4f} Normalized Average loss:{:.4f}'.format(epoch + 1, n_epochs, val_loss, norm_val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        

        # save model
        score = sum(metric.value() for metric in metrics)
        score = score/len(metrics)
        
        message += "\tScore:{}".format(score)
        print(message)

        if (best_score == -1) or (score < best_score):
            best_score = score
            torch.save(model, save_model.split('.')[0]+"_bestscore_"+str(epoch+1)+"."+save_model.split('.')[-1])
            print("Save best score model")

        if (best_train_los == -1) or (train_loss < best_train_los):
            best_train_los = train_loss
            torch.save(model, save_model.split('.')[0]+"_besttrainloss_"+str(epoch+1)+"."+save_model.split('.')[-1])
            print("Save best loss on Train model")
        
        if (best_val_los == -1) or (val_loss < best_val_los):
            best_val_los = val_loss
            torch.save(model, save_model.split('.')[0]+"_bestvalloss_"+str(epoch+1)+"."+save_model.split('.')[-1])
            print("Save best loss on Validation model")

        if save_each_epoch:
            torch.save(model, save_model.split('.')[0]+"_"+str(epoch+1)+"_."+save_model.split('.')[-1])
        
        scheduler.step()


def train_epoch(train_loader, model, out_features, loss_fn, optimizer, cuda_id, log_interval, metrics, max_batchsize=0):
    for metric in metrics:
        metric.reset()
 
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, all_data in enumerate(train_loader):
        if len(all_data)==2:
            (data, target) = all_data
        if len(all_data)==3:
            (data, target, _) = all_data

        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        if max_batchsize>0:
            data = data[:max_batchsize]
            target = target[:max_batchsize]
            
        if cuda_id is not None:
            if len(cuda_id) > 1:
                if target is not None:
                    target = target.cuda()
            else:
                if target is not None:
                    target = target.cuda(cuda_id[0])
        
        
        # if image are not all same size we cannot create a single batch
        if len(data) == 1:        
            if cuda_id is not None:
                if len(cuda_id) > 1:
                    data = tuple(d.cuda() for d in data)
                else:
                    data = tuple(d.cuda(cuda_id[0]) for d in data)
                    
            optimizer.zero_grad()
            outputs = model(*data)
        else:
            # for image inbatch,compute input, appends to out, convert out in tensor
            outputs = torch.zeros(len(data), out_features)
            i=0
            for image_tensor in data:
                image_tensor = image_tensor[None, :]
                if cuda_id is not None:
                    if len(cuda_id) > 1:
                        image_tensor = image_tensor.cuda()
                    else:
                        image_tensor = image_tensor.cuda(cuda_id[0])

                optimizer.zero_grad()
                out = model(image_tensor)
                if type(out) is dict:
                    out = out['phoc'][-1]
                outputs[i,:] = out

                i += 1
        
        torch.cuda.empty_cache()


        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, out_features, loss_fn, cuda_id, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, all_data in enumerate(val_loader):
            if len(all_data)==2:
                (data, target) = all_data
            if len(all_data)==3:
                (data, target, _) = all_data

            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            if cuda_id is not None:
                if len(cuda_id) > 1:
                    data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()
                else:
                    data = tuple(d.cuda(cuda_id[0]) for d in data)
                    if target is not None:
                        target = target.cuda(cuda_id[0])
             
            if cuda_id is not None:
                if len(cuda_id) > 1:
                    if target is not None:
                        target = target.cuda()
                else:
                    if target is not None:
                        target = target.cuda(cuda_id[0])

            # if image are not all same size we cannot create a single batch
            if len(data) == 1:        
                if cuda_id is not None:
                    if len(cuda_id) > 1:
                        data = tuple(d.cuda() for d in data)
                    else:
                        data = tuple(d.cuda(cuda_id[0]) for d in data)
                        
                outputs = model(*data)
            else:
                # for image inbatch, vompute input, appends to out, convert out in tensor
                outputs = torch.zeros(len(data), out_features )
                i=0
                for image_tensor in data:
                    image_tensor = image_tensor[None, :]
                    if cuda_id is not None:
                        if len(cuda_id) > 1:
                            image_tensor = image_tensor.cuda()
                        else:
                            image_tensor = image_tensor.cuda(cuda_id[0])

                    out = model(image_tensor)
                    if type(out) is dict:
                        out = out['phoc'][-1]
                    outputs[i,:] = out

                    i += 1

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics