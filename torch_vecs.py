import torch
import torch.nn as nn
import pandas as pd
import sys


class WMMatrix(nn.Module):
    # Weight matrix object encoding the binding between lexical items and positions
    
    def __init__(self, i_dim, p_dim):
        super().__init__()
        self.current_matrix = nn.Parameter(torch.randn(i_dim, p_dim))
        self.weight = nn.Parameter(torch.tensor(5.0)) 
        self.i_dim = i_dim
        self.p_dim = p_dim

    def encodeMemory(self, i, p):
        # Encode new binding between item vector and position vector
        outer_product = torch.einsum('i,j->ij', i, p)
        with torch.no_grad(): 
          self.current_matrix.data += self.weight * outer_product 

        return self.current_matrix

    def retrieveMemory(self, p_vec):    
        # Retrieve item vector given a position vector 
        retrieved_i = self.current_matrix @ p_vec

        return retrieved_i

def normVec(vec):
    # Normalize vector 
    fvec = vec.float()
    norm = torch.linalg.norm(fvec)
    unit_vec = vec / norm
    type(unit_vec)

    return unit_vec

def perpendicVec(vec):
  # Generate random vector perpendicular to input vector
  random_vector = torch.randn_like(vec)
  vec_squared = torch.dot(vec, vec)

  if vec_squared == 0:
    # If v is a zero vector, any vector is perpendicular.
    # You might return the random_vector or handle as appropriate
    perpendicular_v = random_vector

  else:
    projection = (torch.dot(random_vector, vec) / vec_squared) * vec
    # The perpendicular component is the random_vector minus its projection onto v
    perpendicular_v = random_vector - projection

  return perpendicular_v

def genVectorsByCos(size, cosine, num_vectors):
   # Generate random vectors with a prespecified cosine similarity
   # 1. Generate a normalized random target vector of size 'size' using 'torch.rand'
   random_target = torch.rand(size)
   norm_target = normVec(random_target)
   vectors = torch.zeros(num_vectors, size)
   vectors[0] = norm_target

   for i in range(1,num_vectors):
      # 2. Create a temporary perpendicular vector of the same size
      perpendic_vec = perpendicVec(norm_target)
      norm_perpendic_vec = normVec(perpendic_vec)

      # 3. Compose the distractor vector based on cosine similarity of to the target.
      # The distractor is the linear combination of the target and the perpendicular
      # vector with coefficients cosine and sine (equals sqrt(1 - cosine^2))
      distractor = norm_target*cosine[i-1] + norm_perpendic_vec*(1 - cosine[i-1]**2)**0.5
      norm_distractor = normVec(distractor)

      # 4. Store the distractor
      vectors[i] = norm_distractor

   return vectors

def composeItems(vectors, agr_size, num_val):
    # Combine vector representations of root and agreement into item vectors 
    complete_vectors = []
    if len(vectors) != len(num_val):
        raise ValueError("Vectors and number values must have the same length.")

    for index, value in enumerate(num_val):

      if value not in [0, 1]:
        raise ValueError(f"Number value at index {index} must be 0 or 1.")

      else:
        agr = torch.full((agr_size,),(value))
        if value == 1:
          agr = normVec(agr)
        complete_vectors.append(torch.cat((vectors[index],agr)))

    return complete_vectors  

def cosineSimilarity(vec1, vec2):
  # Compute cosine similarity 
  dot_product = torch.dot(vec1, vec2)
  norm_vec1 = torch.linalg.norm(vec1)
  norm_vec2 = torch.linalg.norm(vec2)

  # Handle cases where one or both vectors are zero:
  if norm_vec1 == 0 or norm_vec2 == 0:
    return torch.tensor(0.0) 
  
  return dot_product / (norm_vec1 * norm_vec2)

def decodeAgreement(retrieved_i, agr_size):
  # Compute probability of responding with a singular or a plural item, respectively 
  retrieved_agr = retrieved_i[-agr_size:] 
  pl = normVec(torch.full((agr_size,),1))
  pl_cosine = max(cosineSimilarity(pl,retrieved_agr),0)
  feature_prob = torch.tensor([1-pl_cosine, pl_cosine])

  return feature_prob

def decodeLexicalRoot(retrieved_i, agr_size, vecs, softmax_temp):
  # For all lexical roots (target and distractors), compute probability of 
  # responding with this root 
  retrieved_sem = retrieved_i[:(len(retrieved_i)-agr_size)]
  similaritiy_scores = torch.zeros(len(vecs))

  for index, vec in enumerate(vecs):
    similaritiy_scores[index] = cosineSimilarity(retrieved_sem, vec)
  softmax_scores = torch.softmax(similaritiy_scores/softmax_temp, dim=0)

  return softmax_scores

def getResponseProb(softmax_scores, feature_prob, num_val):
  # Compute a matrix with probabilities of responding with a singular or plural
  # version of each of the lexical roots
  response_prob = torch.zeros(len(softmax_scores),2)
  for i in range(len(softmax_scores)):
    if num_val[0] == 0:
      response_prob[i] = softmax_scores[i]*feature_prob
    else: 
      reverse_feature_prob = torch.tensor([feature_prob[1], feature_prob[0]])
      response_prob[i] = softmax_scores[i]*reverse_feature_prob

  return response_prob


def main():
  if len(sys.argv) != 8:
     print("Usage: torch_vecs.py softmax_temp root_size root_cos agr_size " \
     "num_vals pos_size pos_cos\n For several distractors, separate cosine values " \
     "by a comma with no intervening space; e.g., 0.1,0.8. Number features of target " \
     "and distractor(s) are encoded as 0 (singular) and 1 (plural). E.g., use 101" \
     " for 'plural, singular, plural'.")

  else:
    # Generate item vectors 
    try:
      root_size = int(sys.argv[2])
    except:
      raise TypeError("Root size needs to be an integer.")
    try:
      root_cos = [float(x) for x in sys.argv[3].split(',')]
    except:
      raise TypeError("Cosine values need to be numerical. For several distractors, " \
      "separate cosines by a comma with no intervening space; e.g., 0.1,0.8.")
    try:
      agr_size = int(sys.argv[4])
    except:
      raise TypeError("Agreement size needs to be an integer.")
    try:
      num_vals = torch.tensor([int(x) for x in list(str(sys.argv[5]))]) 
    except:
      raise TypeError("Number features of target and distractor(s) are encoded " \
      "as 0 (singular) and 1 (plural). E.g., use 101 for 'plural, singular, plural'.")

    root_vecs = genVectorsByCos(root_size, root_cos, len(num_vals)) 
    item_vecs = composeItems(root_vecs, agr_size, num_vals) 

    # Generate position vectors
    try:
      pos_size = int(sys.argv[6])
    except:
      raise TypeError("Position size needs to be an integer.")
    try:
      pos_cos = [float(x) for x in sys.argv[7].split(',')]
    except:
      raise TypeError("Cosine values need to be numerical. For several distractors, " \
      "separate cosines by a comma with no intervening space; e.g., 0.1,0.8.")
    pos_vecs = genVectorsByCos(pos_size, pos_cos, len(num_vals))

    # Encode weight matrix 
    matrix = WMMatrix(root_size + agr_size, pos_size)
    for i in range(len(num_vals)):
      matrix.encodeMemory(item_vecs[i], pos_vecs[i])  

    # Retrieve target item vector from matrix and compute probabilities 
    try:
      softmax_temp = float(sys.argv[1])
    except:
      raise TypeError("Softmax temperature needs to be numerical.")
    retrieved = matrix.retrieveMemory(pos_vecs[0])
    dec_agr = decodeAgreement(retrieved, agr_size)
    dec_lex = decodeLexicalRoot(retrieved, agr_size, root_vecs, softmax_temp)
    prob = getResponseProb(dec_lex, dec_agr, num_vals)
    print(prob)

    # Save to output 
    df = pd.DataFrame(prob.detach().numpy(), columns=['Veridical', 'Nonveridical'])
    df.to_csv('vec_probs.csv')

if __name__ == "__main__":
    main()


