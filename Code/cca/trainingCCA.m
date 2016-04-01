function trainingCCA()

disp('reading data from files')
images = dlmread('images_pert.txt');
sentences = dlmread('sentences_pert.txt');
size(images)
size(sentences)
disp('number of coefficients')
min(rank(images),rank(sentences))
[A,B] = canoncorr(images, sentences);
size(A)
size(B)
dlmwrite('imageprojection_pert.txt',A);
dlmwrite('sentenceprojection_pert.txt',B);
end
