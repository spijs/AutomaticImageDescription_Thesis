function trainingCCA()

images = dlmread('images.txt');
sentences = dlmread('sentences.txt');
size(images)
size(sentences)
disp('number of coefficients')
min(rank(images),rank(sentences))
[A,B] = canoncorr(images, sentences);
size(A)
size(B)
dlmwrite('imageprojection.txt',A);
dlmwrite('sentenceprojection.txt',B);
end