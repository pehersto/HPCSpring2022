function parmatlab()

N = 1e+6;
n = 1e+2;
A = randn(n, n);
B = randn(n, n);

tic
for i=1:N
  C = A*B;
end
toc

% start pool
parfor i=1:20
  x = 3;
end

tic
parfor i=1:N
  C = A*B;
end
toc

end
