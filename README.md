# Test-2
semi_param_est = []
epoch_history = []
test_loss_history = []


g, m = init(num_features, plot=True)

for i in range(MC_no):

  # Create Data Set
  train_dataset = constant_effect_data(theta, sample_size, num_features, g, m)
  test_dataset = constant_effect_data(theta, sample_size, num_features, g, m)

  # Create Model
  model = semi_param_net(num_features, hidden_depth, hidden_width, "semi_model")
  model.to(ptu.device) # Move model to device
  
  # Results
  train_args = dict(num_cross_fits = num_cross_fits, batch_size = batch_size, num_epochs = num_epochs, lr = lr, early_stop = early_stop, tt_convergence = tt_convergence)  
  results = train_epochs(model, train_dataset, test_dataset, train_args, i)
  
  if i>0:
    semi_param_est.append(results[0])
    test_loss_history.append(results[1])
    epoch_history.append(results[2])
    
    
  else:
    semi_param_est.append(results[0])
    test_loss_history.append(results[1])
    # Plot Variables
    initial_train_loss = results[2]
    initial_valid_loss = results[3]
    estimates_history = results[4]
    epoch_history.append(results[5])
