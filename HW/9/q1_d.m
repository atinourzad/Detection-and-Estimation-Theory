clc
clear

sigma = 10 * eye(3);
F = [0 0.8 1.5; 0 0 0; 0 1 0];
G = [1; 1; 0];
H = [1; 0; 0];

Q = 1;
S = 0;
R = 0.1;

for i=1:32
   k = ( F*sigma*H+ G*S ) * inv(H'*sigma*H+R);
   sigma = (F-k*H')*sigma*((F-k*H')') + [G -k]*[Q S; S R]*[G'; -k'];
end

fprintf('After 32 iterations sigma is: \n');
disp(sigma);

MSE = [1 0 0]*sigma*[1; 0; 0];
fprintf('The resulting MSE is: %f \n', MSE);