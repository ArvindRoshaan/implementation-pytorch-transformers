import numpy as np
import torch

#the dimension of all intermediate outputs in the model
d_model = 512
#the number of attention heads
h = 8

def softmax(array):
    """
    Applies softmax function along the rows of the array of dimension (n, m) 
    Args:
        array (torch.Tensor): torch.Tensor with dimension (n, m)
    Returns:
        torch.Tensor: Output tensor of dimension (n, m)
    """
    return (torch.exp(array).T/torch.sum(torch.exp(array), axis=1)).T
  
def single_head_attention(query, key, value):
    """
    Perform forward single head attention operation on the given queries, keys and values
    Args:
        query (torch.Tensor): torch.Tensor with dimension (d_model, d_k), where d_k is the dimension of queries
        key (torch.Tensor): torch.Tensor with dimension (d_model, d_k), where d_k is the dimension of keys
        value (torch.Tensor): torch.Tensor with dimension (d_model, d_v), where d_v is the dimension of values
    Returns:
        torch.Tensor: torch.Tensor with dimension (d_model, d_v)
    """
    d_k = key.shape[0]
    return torch.matmul(softmax(torch.matmul(query, key.T)/np.power(d_k, 0.5)), value)
  
class multi_head_attention:
    """
    Performs multi head attention
    Attributes:
        head_count (positive integer): #the number of attention heads
        d_model (positive integer): #the output dimension of all sub-layers in the model
        d_k (positive integer): #the dimension of keys and queries
        d_v (positive integer): #the dimension of values
        W_heads (dict): dictionary of dictionaries for each head, containing weight matrices of queries, keys and values
                        index by "Q", "K" and "V" respectively
        W_concat (dict): weight matrix of dimension (head_count*d_v, d_model)
    Methods:
        feed_forward(array, non_lin_fn, non_lin_fn_name=None, non_lin_fn_param=0.1)
            Perform feed-forward operation from input to output in CNN
    """
    def __init__(self, n_heads, d_model):
        self.head_count = n_heads
        self.d_model = d_model
        self.d_k = d_model//self.head_count
        self.d_v = d_model//self.head_count
        self.W_heads = dict()
        for i in range(self.head_count):
            self.W_heads[i] = dict()
            self.W_heads[i]["Q"] = torch.rand(self.d_model, self.d_k, requires_grad=True)
            self.W_heads[i]["K"] = torch.rand(self.d_model, self.d_k, requires_grad=True)
            self.W_heads[i]["V"] = torch.rand(self.d_model, self.d_v, requires_grad=True)
        self.W_concat = torch.rand(self.head_count*self.d_v, self.d_model, requires_grad=True)
    
    
    def forward(self, queries, keys, values):
        """
        Perform forward multi head attention operation on the given queries, keys and values
        Args:
            queries (torch.Tensor): torch.Tensor with dimension (self.d_model, self.d_model)
            keys (torch.Tensor): torch.Tensor with dimension (self.d_model, self.d_model)
            values (torch.Tensor): torch.Tensor with dimension (self.d_model, self.d_model)
        Returns:
            torch.Tensor: torch.Tensor with dimension (self.d_model, self.d_model)
        """
        concat_heads = torch.zeros((self.d_model, self.d_model))
        for i in range(self.head_count):
            query = torch.matmul(queries, self.W_heads[i]["Q"])
            key = torch.matmul(keys, self.W_heads[i]["K"])
            value = torch.matmul(values, self.W_heads[i]["V"])
            head = single_head_attention(query, key, value)
            concat_heads[:, i*self.d_v : (i+1)*self.d_v ] = head
        return torch.matmul(concat_heads, self.W_concat)

