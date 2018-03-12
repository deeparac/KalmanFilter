N = 50;

% initialize real X
X1 = zeros(1, N);
X2 = zeros(1, N);
X1(1) = 1;
X2(1) = 1;
X = [X1; X2];

% this is for update step, ie, X_t|t
X_kalman = zeros(2, N);
X_kalman(1:2, 1) = [1; 1];

% system state Z
Z = zeros(1, N);

% error covariance lambda_X_t|t
P = zeros(2, 2, N);
P(1:2, 1:2, 1) = [20 0; 0 20];

% one step prediction error, ie, lambda_X_t+1|t
PList = zeros(2, 2, N);

% Kalman Gain
K = zeros(2, N);

% noise V
V = [randn(1, N) * sqrt(20)];

% x[n+1] = A * x[n]
A = [1 1; 0 1];

% z[n+1] = C * x[n+1] + v[n+1]
C = [1 0];

% prediction step X_t+1|t
XList = zeros(2, N);

for i = 1 : N-1
    
    % state model
    X(1:2, i+1) = A * X(1:2, i);
    Z(1, i+1) = C * X(1:2, i+1) + V(1, i+1);

    % calculate prediction
    Xpred = A * X_kalman(1:2, i);
    Ppred = A * P(1:2, 1:2, i) * A';
    PList(1:2, 1:2, i) = Ppred;
    XList(1:2, i) = Xpred;
    
    % calculate Kalman Gain
    K(1:2, i+1) = Ppred * C' * inv(C * Ppred * C' + 20);
    
    % calculate update
    X_kalman(1:2, i+1) = K(1:2, i+1) * (Z(1, i+1) - C * Xpred) + Xpred;
    P(1:2, 1:2, i+1) = (eye(2) - K(1:2, i+1) * C) * Ppred;
    
end

% sm -> smooth
L = zeros(2, 2, N);
X_sm = zeros(2, N);
P_sm = zeros(2, 2, N);

X_sm(1:2, N) = XList(1:2, N-1);
P_sm(1:2, 1:2, N) = PList(1:2, 1:2, N-1);

for k = N-1 : -1 : 1
    L(1:2, 1:2, k) = P(1:2, 1:2, k) * A' * inv(PList(1:2, 1:2, k));
    X_sm(1:2, k) = X_kalman(1:2, k) + ...
        L(1:2, 1:2, k) * (X_sm(1:2, k+1) - XList(1:2, k));
    P_sm(1:2, 1:2, k) = P(1:2, 1:2, k) + ...
        L(1:2, 1:2, k) * (P_sm(1:2, 1:2, k+1) - PList(1:2, 1:2, k+1)) * ...
        L(1:2, 1:2, k)';
end

e1 = zeros(1, N);
e2 = zeros(1, N);
for i = 1 : N
    e1(i) = P(1, 1, i);
    e2(i) = P(2, 2, i);
end

e3 = zeros(1, N);
e4 = zeros(1, N);
for i = 1 : N
    e3(i) = P_sm(1, 1, i);
    e4(i) = P_sm(2, 2, i);
end

figure;
subplot(511);
plot(1:N, X(1, 1:N));
hold on;
plot(1:N, Z);
hold off;
legend('X_1[n]', 'Z[n]');
title('(1)');

subplot(512);
plot(1:N, e1(1, 1:N));
hold on;
plot(1:N, e3(1, 1:N));
legend('lambda_e_1filter[n]', 'lambda_e_1smoother[n]');
title('(2)');

subplot(513);
plot(1:N, e2(1, 1:N));
hold on;
plot(1:N, e4(1, 1:N));
legend('lambda_e_2filter[n]', 'lambda_e_2smoother[n]');
title('(3)');

subplot(514);
plot(1:N, X_kalman(1, 1:N));
hold on;
plot(1:N, X_sm(1, 1:N));
hold off;
legend('X_1filter[n]', 'X_1smoother[n]');
title('(4)');

subplot(515);
plot(1:N, X_kalman(2, 1:N));
hold on;
plot(1:N, X_sm(2, 1:N));
hold off;
legend('X_2filter[n]', 'X_2smoother[n]');
title('(5)');
