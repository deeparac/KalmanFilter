N = 200;

A = [0.9 0; 0 0.9];

w = randn(2, N);
v = randn(2, N);

Q = [1 0; 0 1];
R = [1 0; 0 1];


B = [eps 0; 0 eps] .^ 2;

C = [1 0; 1 eps];

X = zeros(2, N);
Y = zeros(2, N);

X(1:2, 1) = B * w(1:2, 1);
Y(1:2, 1) = v(1:2, 1);

for t = 1 : N-1
    X(1:2, t+1) = A * X(1:2, t) + B * w(1:2, t);
    Y(1:2, t+1) = C * X(1:2, t+1) + v(1:2, t+1);
end

% Kalman Filter Here!

% Initialization
    P = zeros(2, 2, N);
    P(1:2, 1:2, 1) = 1/eps .* eye(2);
    Pkf = zeros(2, 2, N);
    Xkf = zeros(2, N);
    K = zeros(2, 2, N);
eps_list = [1 2.2204e-16 1e-16];
for k = 1 : 3
    eps = eps_list(k);
    for i = 1 : N-1

        % Prediction
        Xkf(1:2, i+1) = A * X(1:2, i);
        Pkf(1:2, 1:2, i+1) = A * P(1:2, 1:2, i) * A' + ...
                            B * Q * B';

        % Kalman Gain
        K(1:2, 1:2, i+1) = Pkf(1:2, 1:2, i+1) * C' / ...
                            (C * Pkf(1:2, 1:2, i+1) * C + ...
                                R);

        % Updated
        Xup(1:2, i+1) = K(1:2, 1:2, i+1) * ...
                        (Y(1:2, i+1) - C * Xkf(1:2, i+1)) + ...
                        Xkf(1:2, i+1);
        P(1:2, 1:2, i+1) = (eye(2) - K(1:2, 1:2, i+1) * C) * ...
                            Pkf(1:2, 1:2, i+1);    
    end  

    e1 = zeros(1, N);
    e2 = zeros(1, N);
    e3 = zeros(1, N);
    e4 = zeros(1, N);
    
    h = 1:N;
    
    for i = h
        e1(i) = P(1, 1, i);
        e2(i) = P(2, 2, i);
        e3(i) = Pkf(1, 1, i);
        e4(i) = Pkf(2, 2, i);
    end

    figure;
    subplot(221);
    plot(h, X(1, h), 'g');
    hold on;
    plot(h, Xup(1, h), 'r');
    hold off;
    legend('original', 'estimation');
    title('x_1[n]');
    subplot(222);
    plot(h, X(2, h), 'g');
    hold on;
    plot(h, Xup(2, h), 'r');
    hold off;
    legend('original', 'estimation');
    title('x_2[n]');
    subplot(223);
    plot(h, e1(h), 'g');
    hold on;
    plot(h, e2(h), 'r');
    hold off;
    legend('estimation errors', 'one-step-prediction errors');
    title('x_1[n]');
    subplot(224);
    plot(h, e3(h), 'g');
    hold on;
    plot(h, e4(h), 'r');
    hold off;
    legend('estimation errors', 'one-step-prediction errors');
    title('x_1[n]');
end    









