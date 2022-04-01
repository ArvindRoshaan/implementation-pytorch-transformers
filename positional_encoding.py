def positionalEncoding(n_pos, d_word_vec):
  """
    positional encoding
    Attributes:
        n_pos (positive integer): #positions
        d_word_vec (positive integer): embedding dimension
    """
  temp1 = []
  for i in range(n_pos):
    temp2 = []
    for j in range(d_word_vec):
      temp2.append(i / np.power(10000, 2 * (j // 2) / d_word_vec))
    temp1.append(np.array(temp2))
  PE = np.array(temp1)
  PE[:, 0::2] = np.sin(PE[:, 0::2])
  PE[:, 1::2] = np.cos(PE[:, 1::2])
  return PE
