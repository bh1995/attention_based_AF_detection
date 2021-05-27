# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:56:04 2021

@author: bjorn
"""


def train_model(model, train_loader, val_loader, n_epochs, optimizer, scheduler, criterion, save_wts=None):
  history = dict(train=[], val=[], accuracy=[])
  true_label = []
  predictions = []
  best_loss = np.inf
  best_model_wts = copy.deepcopy(model.state_dict())
  # widths = np.arange(0.01, 10, 0.01) 
  widths = np.arange(0.01, 50, 0.1) 
  # widths = np.arange(1, 301, 0.01)
  # accuracy = []
  for epoch in tqdm(range(1, n_epochs+1), desc='Training Epoch loop'):
    model = model.train()
    train_losses = []
    # for batch in tqdm(train_loader, mininterval=0.5, desc='- Training Batch loop', leave=False):
    for batch in train_loader:
      # Perform training
      # sig, label = map(lambda x: x.to(device), batch)
      sig, label = batch
      # fs_in = 3000
      # fs_out = 1000
      # secs = len(sig[0,:,0])/fs_in # Number of seconds in signal record
      # samps = secs*fs_out   # Number of samples to downsample
      # sig = signal.resample(sig[0,:,0], num=int(samps)) # resample signal to correct fs_out
      # sig = (sig-np.min(sig))/np.ptp(sig) # nromalize [0,1]
      # sig = sig[None,None,:,:]
      rri = get_rri(sig)
      sig, rri, label = torch.tensor(sig, dtype=torch.float, device=device), torch.tensor(rri, dtype=torch.float, device=device), torch.tensor(label, dtype=torch.float, device=device)
      # zero optimizer gradients, forward pass, compute loss, backward pass, compute optimizer step 
      optimizer.zero_grad()
      output = model(sig, rri)
      # output = torch.argmax(nn.functional.softmax(output))
      # print(output, nn.functional.softmax(output))
      # print(output)
      # print('output:',output.shape, 'label:', label.shape)
      loss = criterion(output[:,0], label)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())

    # Perfrom inference
    model = model.eval()
    val_losses = []
    true_label = np.array([])
    predictions = np.array([])
    with torch.no_grad():
      for batch in val_loader:
        sig, label = batch
        # fs_in = 3000
        # fs_out = 1000
        # secs = len(sig[0,:,0])/fs_in # Number of seconds in signal record
        # samps = secs*fs_out   # Number of samples to downsample
        # sig = signal.resample(sig[0,:,0], num=int(samps)) # resample signal to correct fs_out
        # sig = signal.cwt(sig[0,:,0], signal.ricker, widths=widths)[:,:]
        # sig = (sig-np.min(sig))/np.ptp(sig) # nromalize [0,1]
        # sig = sig[None,None,:,:]
        rri = get_rri(sig)
        sig, rri, label = torch.tensor(sig, dtype=torch.float, device=device), torch.tensor(rri, dtype=torch.float, device=device), torch.tensor(label, dtype=torch.float, device=device)
        output = model(sig, rri)
        # output = torch.sigmoid(output[:,0])
        loss = criterion(output[:,0], label)
        val_losses.append(loss.item())
        # predictions.append(np.round(output.cpu().detach()))
        # true_label.append(np.round(label.cpu().detach()))
        # predictions = np.append(predictions, torch.argmax(nn.functional.softmax(output[0])).cpu().detach())
        predictions = np.append(predictions, np.round(torch.sigmoid(output[:,0]).cpu().detach()))
        true_label = np.append(true_label, label.cpu().detach())  
      # Get losses and accuracy
      cm = confusion_matrix(true_label, predictions, labels=[0,1])
      acc = np.sum(np.diag(cm))/np.sum(cm)
      # pred = []
      # for sublist in predictions:
      #   for i in sublist:
      #     pred.append(i.cpu().numpy())
      # true_lab = []
      # for sublist in true_label:
      #   for i in sublist:
      #     true_lab.append(i.cpu().numpy())
      # cm = confusion_matrix(true_lab, pred)
      # # print(cm)
      # acc = (cm[0,0]+cm[1,1])/cm.sum()
      # accuracy.append(acc)

      print('\n Train Loss:', np.mean(train_losses))
      print(' Val Loss:', np.mean(val_losses))
      print(' Accuracy on validation set:', np.round(acc, 5))
      # print('Confusion Matrix: \n', cm)
      plot_confusion_matrix(cm, ['AF', 'Normal'])

      
    train_loss = np.mean(train_losses)  
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    history['accuracy'].append(acc)

    # perform learning rate scheduler step before next epoch
    if epoch > 35:
      scheduler.step()

    # Save best model weights
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    
    # Save weights after given epoch
    if save_wts != None:
      if epoch%10 == 0: # save wts every 20th epoch
        model_save = model
        model_save.load_state_dict(best_model_wts)
        model_save = model_save.eval()
        torch.save(model_save, save_wts)
        del model_save
        # plot training loss
        ax = plt.figure().gca()
        ax.plot(history['train'])
        ax.plot(history['val'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'])
        plt.title('Loss over training epochs')
        plt.show();
        print('\n')
        # Plot accuracy
        ax = plt.figure().gca()
        ax.plot(history['accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Validation set'])
        plt.title('Validation accuracy over epochs')
        plt.show();

  model.load_state_dict(best_model_wts)
  return model.eval(), history


