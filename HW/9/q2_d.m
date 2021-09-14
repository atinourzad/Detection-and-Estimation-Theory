clc
clear

sigma = 10 * eye(2);
F = [0.3 0.18; 1 0];
G = [1; 0];
H = [0.3; 0.18];

Q = 1.44;
S = 1.44;
R = 1.54;

for i=1:4
   k = ( F*sigma*H+ G*S ) * inv(H'*sigma*H+R);
   sigma = (F-k*H')*sigma*((F-k*H')') + [G -k]*[Q S; S R]*[G'; -k'];
end

fprintf('After 4 iterations sigma is: \n');
disp(sigma);

MSE = [1 0]*sigma*[1;0];
fprintf('The resulting MSE is: %f \n', MSE);