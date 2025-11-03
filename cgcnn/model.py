from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.nbr_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 self.nbr_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
        self.softplus3 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        
        N, M = nbr_fea_idx.shape   
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  
        total_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        
        #for atom:
        total_gated_fea = self.fc_full(total_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        
        ato_fea = self.softplus2(atom_in_fea + nbr_sumed)

        #for environment:

        nbr_fea_update = self.nbr_full(total_fea)
        nbr_fea_update = self.softplus3(nbr_fea_update+nbr_fea)

        return ato_fea,nbr_fea_update


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties. Modified for multi-task learning (predicting work function and band gap).
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, pooling_method='mean'):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        pooling_method: str
          Method used for pooling atom features to crystal features ('mean' or 'attention')
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.pooling_method = pooling_method  # 
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_one_degree = nn.Linear(h_fea_len, 1)
        
        if self.pooling_method == 'attention':
            self.attention_fc = nn.Linear(h_fea_len, 1)  # 
            self.attention_softplus = nn.Softplus()
        
        # Shared hidden layers
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        # Two output heads for dual-task learning (work function and band gap)
        if self.classification:
            self.fc_out_task1 = nn.Linear(h_fea_len, 2)  # Task 1: Work function
            self.fc_out_task2 = nn.Linear(h_fea_len, 2)  # Task 2: Band gap
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out_task1 = nn.Linear(h_fea_len, 1)  # Task 1: Work function
            self.fc_out_task2 = nn.Linear(h_fea_len, 1)  # Task 2: Band gap
        
        self.fc_out_material_type = nn.Linear(h_fea_len, 2)
        self.material_type_sigmoid = nn.Sigmoid()
        
        self.h_fea_len = h_fea_len
        
        self._aux_components_initialized = False

    def _initialize_aux_components(self):
        """
        """
        if not self._aux_components_initialized:
            self.fc_out_aux_task1 = nn.Linear(self.h_fea_len, 1)  
            self.fc_out_aux_task2 = nn.Linear(self.h_fea_len, 1)  
            self.fc_out_aux_task3 = nn.Linear(self.h_fea_len, 1)  
            
            nn.init.normal_(self.fc_out_aux_task1.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.fc_out_aux_task2.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.fc_out_aux_task3.weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.fc_out_aux_task1.bias)
            nn.init.zeros_(self.fc_out_aux_task2.bias)
            nn.init.zeros_(self.fc_out_aux_task3.bias)
            
            if next(self.parameters()).is_cuda:
                self.fc_out_aux_task1 = self.fc_out_aux_task1.cuda()
                self.fc_out_aux_task2 = self.fc_out_aux_task2.cuda()
                self.fc_out_aux_task3 = self.fc_out_aux_task3.cuda()
            
            self._aux_components_initialized = True
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, use_auxiliary_tasks=False):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        use_auxiliary_tasks: bool
          Whether to compute and return auxiliary task outputs

        Returns
        -------

        prediction_task1: nn.Variable shape (N0, 1)
          Prediction for work function
        prediction_task2: nn.Variable shape (N0, 1)
          Prediction for band gap
        prediction_material_type: nn.Variable shape (N0, 2)
          Prediction for material type (probability of being semiconductor)
        atom_features_list: List of atom features for each crystal
        aux_outputs: tuple of (aux_output1, aux_output2, aux_output3) or None
          Auxiliary task outputs if use_auxiliary_tasks is True
        """
        if use_auxiliary_tasks and not self._aux_components_initialized:
            self._initialize_aux_components()
            
        
        atom_fea = self.embedding(atom_fea)  #
        for conv_func in self.convs:
            atom_fea,_ = conv_func(atom_fea, nbr_fea, nbr_fea_idx) 
        atom_fea = self.conv_to_fc(atom_fea)  #
        atom_fea = self.conv_to_fc_softplus(atom_fea)  #
        
        # Store features at this point for atom contributions
        h_fea = atom_fea

        # Get atom-level scores for visualization/interpretation
        atom_fea = self.conv_one_degree(atom_fea) #N*1

        atom_features_list = []
        atom_features_list = self.get_crystal_atom_features(atom_fea, crystal_atom_idx)

        # Get crystal-level representation through pooling
        crys_fea = self.pooling(h_fea, crystal_atom_idx) #
        
        # Apply additional hidden layers if specified
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

      
        material_type_logits = self.fc_out_material_type(crys_fea)
        material_type_probs = torch.softmax(material_type_logits, dim=1)

        # 
        if self.classification:
            crys_fea = self.dropout(crys_fea)
            pred_task1 = self.logsoftmax(self.fc_out_task1(crys_fea))
            pred_task2 = self.logsoftmax(self.fc_out_task2(crys_fea))
        else:
            pred_task1 = self.fc_out_task1(crys_fea)
            pred_task2 = self.fc_out_task2(crys_fea)
            
        
        aux_outputs = None
        if use_auxiliary_tasks:
            
            aux_output1 = self.fc_out_aux_task1(crys_fea)  
            aux_output2 = self.fc_out_aux_task2(crys_fea) 
            aux_output3 = self.fc_out_aux_task3(crys_fea)  
            aux_outputs = (aux_output1, aux_output2, aux_output3)
        
        return pred_task1, pred_task2, material_type_probs, atom_features_list, aux_outputs

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        if self.pooling_method == 'mean':
            summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                          for idx_map in crystal_atom_idx]
            return torch.cat(summed_fea, dim=0)
        elif self.pooling_method == 'attention':
            # Attention pooling implementation
            attention_weights = [self.attention_softplus(self.attention_fc(atom_fea[idx_map]))
                                 for idx_map in crystal_atom_idx]
            attention_weights = [torch.softmax(weights, dim=0) for weights in attention_weights]
            pooled_fea = [torch.sum(weights * atom_fea[idx_map], dim=0, keepdim=True)
                          for weights, idx_map in zip(attention_weights, crystal_atom_idx)]
            return torch.cat(pooled_fea, dim=0)
        else:
            raise ValueError("Invalid pooling method. Choose 'mean' or 'attention'.")
    
    def get_crystal_atom_features(self, atom_fea, crystal_atom_idx):
      """
      Get the features for each atom in each crystal without changing the original output.

    Parameters
    ----------
    atom_fea: Variable(torch.Tensor) shape (N, 1)
      Atom feature vectors of the batch
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx

    Returns
    -------
    list of torch.Tensor
      A list where each element is a tensor representing the features of atoms in a crystal.
      """
    # Create an empty list to store the features for each crystal
      crystal_atom_features_list = []
    
    # Iterate over each crystal
      for idx_map in crystal_atom_idx:
        # Select the features for the atoms in the current crystal
          crystal_atom_features = atom_fea[idx_map]
        
        # Append the features to the list
          crystal_atom_features_list.append(crystal_atom_features.view(-1))
    
      return crystal_atom_features_list

