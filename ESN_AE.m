clear all
close all
clc
% seed random generator
tic
rand('state', sum(100*clock));

donTr=load('ECG200_TRAIN');
donTs=load('ECG200_Test');
don=[donTr(:,2:end);donTs(:,2:end)];
don=don';
donT=[donTr(:,1);donTs(:,1)];

% unit counts (input, hidden, output)
IUC = 96;
HUC = 300;
OUC = 96;

IPP=don;
TPP=IPP;

probInp  = [  1.00 ];
rngInp   = [  1 ]; 
probRec  = [  0.01];
rngRec   = [ -0.6 ];
probBack = [ 0.0  ];
rngBack  = [0.0 ];
w_in = zeros(HUC, IUC, length(probInp));
w_rec = zeros(HUC, HUC, length(probRec));
%w_back = zeros(HUC, OUC, length(probBack));

for d=(1:length(probInp))
    w_in(:,:,d) = init_weights(w_in(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probRec))
    w_rec(:,:,d) = init_weights(w_rec(:,:,d), probRec(d),rngRec(d));
end;

% for d=(1:length(probBack))
%     w_back(:,:,d) = init_weights(w_back(:,:,d), probBack(d),rngBack(d));
% end;
SpecRad = max(abs(eig(w_rec(:,:,1))));
if SpecRad>0,
    w_rec = w_rec ./ SpecRad;
end
SpecRad;

x = zeros(HUC,size(TPP,2));
x(:,1) = rand(1,HUC);
w_out=rand(OUC,HUC);
for t=2:size(TPP,2), %run without any learning/training in reservoir and readout unit
    x(:,t) = tanh(w_in*IPP(:,t) + w_rec*x(:,t-1));  
end
%plot(x);
w_out = TPP(:,3:end)*pinv(x(:,3:end));
% w_in=w_out';
% for t=2:size(TPP,2), %run without any learning/training in reservoir and readout unit
%     
%     x(:,t) = tanh(w_in*IPP(:,t) + w_rec*x(:,t-1));
%     
% end

IP=x(:,1:100)';
TP=donT(1:100,:);
IPT=x(:,101:end)';
TPT=donT(101:end,:);
svmStruct = svmtrain(IP,TP);
Group = svmclassify(svmStruct,IPT);
SVMStruct1 = svmtrain(IP,TP);
Group1 = svmclassify(SVMStruct1,IPT);
RD=0; 
for(z=1:size(TPT,1));
if (Group1(z,:) == TPT(z,:))
    RD=RD+1;
end
end
  precision= RD/size(TPT,1)
