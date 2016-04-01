function createSubmatrices()
  projection = dlmread('imageprojection.txt', ',');
  for k = [128 256 512 1024]
    submatrix = projection(:,1:k);
    filename = strcat('imageprojection_',int2str(k),'_pert.txt');
    dlmwrite(filename, submatrix);
  end

end
