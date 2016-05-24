% read a matrix from disk and create 4 submatrices, based on an array of sizes.
function createSubmatrices()
  projection = dlmread('imageprojection_pert.txt', ',');
  for k = [128 256 512 1024]
    submatrix = projection(:,1:k); % take the k first columns of the matrix
    filename = strcat('imageprojection_',int2str(k),'_pert.txt');
    dlmwrite(filename, submatrix);
  end

end
