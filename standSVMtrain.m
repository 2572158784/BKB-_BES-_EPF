function [alphaStar, bStar, SVIndex] = standSVMtrain(n,Y, C, ker, p1)

% Input:
% X: num x dim Data, X has num points, every point is described by dim vectors
% Y: n x 1 Data, {1, -1}
% C: penalty
% kernel: see yxcSVMkernel
% sigma: see yxcSVMkernel
% Output:
% alphaStar: Optimized alpha, see page 193
% bStar: bias, see page 193
% SVIndex: Index for support vectors
Y = Y(:);
H = (Y*Y').*ker;
H=(H+H')/2;
f =-ones(n, 1);
A = zeros(1, n);
%A=[];
b = [];
Aeq = Y';
beq = 0;
lb = zeros(n, 1);
ub = C * ones(n, 1);
x0 = zeros(n,1);
%qp_options = optimset('Display','off');
qp_options = optimset('MaxIter',10^3, 'LargeScale', 'on', 'Display','off');
[alphaStar,fval,exitflag,output] = quadprog(H,f,[],[],Aeq,beq,lb,ub,x0,qp_options);

nearZero = 10^-12;
% Assume the minor ones are all zero.
alphaStar(find(abs(alphaStar) < nearZero)) = 0;
alphaStar(find(alphaStar > C - nearZero)) = C;
% support vectors are those whose alpha value > 0
SVIndex = find(alphaStar > 0);%????????????
% support vectors on bound are those whose alpha value == max(alphaStar) == C
SVNotOnBoundIndex = find(alphaStar > 0 & alphaStar < max(alphaStar));
 
% bStar is an average value, not a random one. 
if ~isempty(SVNotOnBoundIndex)
    bStar = sum( Y(SVNotOnBoundIndex) -  H(SVNotOnBoundIndex, SVIndex) * alphaStar(SVIndex) .* Y(SVNotOnBoundIndex) )...
                 / length(SVNotOnBoundIndex);
else
    bStar = 0;
end
