%% Firefly Fuzzy Regression Algorithm - Created in (9 Jan 2022).
% Firefly algorithm is one of the most decent algorithms in optimization
% which could be used for various tasks and biasing weights. It is
% relatively faster than others just like DE algorithm. So, there was no a
% proper evolutionary linear regression Matlab code available in the web
% and I decided to make one. You can use your data. System uses fuzzy logic
% to create initial model and biases weights into a fit model by Firefly algorithm. You can play
% with parameters (higher values, more calculation time).
% ------------------------------------------------ 
% Feel free to ontact me if you find any problem using the code: 
% mosavi.a.i.buali@gmail.com 
% SeyedMuhammadHosseinMousavi 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916# 
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)

%% Well, Lets do it !!!
clc;
clear;
warning('off');
% Data Loading
data=JustLoad();
% Generate Fuzzy Model
ClusNum=4; % Number of Clusters in FCM
%
fis=GenerateFuzzy(data,ClusNum);
%
%% Tarining FireFly Algorithm
FireFlyFis=FireFlyRegression(fis,data);        

%% Plot Fuzzy FireFly Results (Train - Test)
% Train Output Extraction
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,FireFlyFis);
% Test Output Extraction
TsTar=data.TestTargets;
TestOutputs=evalfis(data.TestInputs,FireFlyFis);
% Train calc
Errors=data.TrainTargets-TrainOutputs;
MSE=mean(Errors.^2);RMSE=sqrt(MSE);  
error_mean=mean(Errors);error_std=std(Errors);
% Test calc
Errors1=data.TestTargets-TestOutputs;
MSE1=mean(Errors1.^2);RMSE1=sqrt(MSE1);  
error_mean1=mean(Errors1);error_std1=std(Errors1);
% Train
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1);
plot(data.TrainTargets,'c');hold on;
plot(TrainOutputs,'k');legend('Target','Output');
title('FireFly Training Part');xlabel('Sample Index');grid on;
% Test
subplot(3,2,2);
plot(data.TestTargets,'c');hold on;
plot(TestOutputs,'k');legend('FireFly Target','FireFly Output');
title('FireFly Testing Part');xlabel('Sample Index');grid on;
% Train
subplot(3,2,3);
plot(Errors,'k');legend('FireFly Training Error');
title(['Train MSE =     ' num2str(MSE) '  ,     Train RMSE =     ' num2str(RMSE)]);grid on;
% Test
subplot(3,2,4);
plot(Errors1,'k');legend('FireFly Testing Error');
title(['Test MSE =     ' num2str(MSE1) '  ,    Test RMSE =     ' num2str(RMSE1)]);grid on;
% Train
subplot(3,2,5);
h=histfit(Errors, 50);h(1).FaceColor = [.8 .8 0.3];
title(['Train Error Mean =   ' num2str(error_mean) '  ,   Train Error STD =   ' num2str(error_std)]);
% Test
subplot(3,2,6);
h=histfit(Errors1, 50);h(1).FaceColor = [.8 .8 0.3];
title(['Test Error Mean =   ' num2str(error_mean1) '  ,   Test Error STD =    ' num2str(error_std1)]);

%% Plot Just Fuzzy Results (Train - Test)
% Train Output Extraction
fTrainOutputs=evalfis(data.TrainInputs,fis);
% Test Output Extraction
fTestOutputs=evalfis(data.TestInputs,fis);
% Train calc
fErrors=data.TrainTargets-fTrainOutputs;
fMSE=mean(fErrors.^2);fRMSE=sqrt(fMSE);  
ferror_mean=mean(fErrors);ferror_std=std(fErrors);
% Test calc
fErrors1=data.TestTargets-fTestOutputs;
fMSE1=mean(fErrors1.^2);fRMSE1=sqrt(fMSE1);  
ferror_mean1=mean(fErrors1);ferror_std1=std(fErrors1);
% Train
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1);
plot(data.TrainTargets,'m');hold on;
plot(fTrainOutputs,'k');legend('Target','Output');
title('Fuzzy Training Part');xlabel('Sample Index');grid on;
% Test
subplot(3,2,2);
plot(data.TestTargets,'m');hold on;
plot(fTestOutputs,'k');legend('Target','Output');
title('Fuzzy Testing Part');xlabel('Sample Index');grid on;
% Train
subplot(3,2,3);
plot(fErrors,'g');legend('Fuzzy Training Error');
title(['Train MSE =     ' num2str(fMSE) '   ,    Test RMSE =     ' num2str(fRMSE)]);grid on;
% Test
subplot(3,2,4);
plot(fErrors1,'g');legend('Fuzzy Testing Error');
title(['Train MSE =     ' num2str(fMSE1) '   ,    Test RMSE =     ' num2str(fRMSE1)]);grid on;
% Train
subplot(3,2,5);
h=histfit(fErrors, 50);h(1).FaceColor = [.3 .8 0.3];
title(['Train Error Mean =    ' num2str(ferror_mean) '   ,   Train Error STD =    ' num2str(ferror_std)]);
% Test
subplot(3,2,6);
h=histfit(fErrors1, 50);h(1).FaceColor = [.3 .8 0.3];
title(['Test Error Mean =    ' num2str(ferror_mean1) '   ,   Test Error STD =    ' num2str(ferror_std1)]);

%% Regression Plots
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
[population2,gof] = fit(TrTar,TrainOutputs,'poly4');
plot(TrTar,TrainOutputs,'o',...
    'LineWidth',1,...
    'MarkerSize',6,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor',[0.9,0.1,0.1]);
    title(['FireFly Train - R =  ' num2str(1-gof.rmse)]);
        xlabel('Train Target');
    ylabel('Train Output');   
hold on
plot(population2,'b-','predobs');
    xlabel('Train Target');
    ylabel('Train Output');   
hold off
subplot(2,2,2)
[population2,gof] = fit(TsTar, TestOutputs,'poly4');
plot(TsTar, TestOutputs,'o',...
    'LineWidth',1,...
    'MarkerSize',6,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor',[0.9,0.1,0.1]);
    title(['FireFly Test - R =  ' num2str(1-gof.rmse)]);
    xlabel('Test Target');
    ylabel('Test Output');    
hold on
plot(population2,'b-','predobs');
    xlabel('Test Target');
    ylabel('Test Output');
 hold off
subplot(2,2,3)
[population2,gof] = fit(TrTar,fTrainOutputs,'poly4');
plot(TrTar,fTrainOutputs,'o',...
    'LineWidth',1,...
    'MarkerSize',6,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.3,0.9,0.2]);
    title(['Fuzzy Train - R =  ' num2str(1-gof.rmse)]);
        xlabel('Train Target');
    ylabel('Train Output');   
hold on
plot(population2,'r-','predobs');
    xlabel('Train Target');
    ylabel('Train Output');   
hold off
subplot(2,2,4)
[population2,gof] = fit(TsTar, fTestOutputs,'poly4');
plot(TsTar, fTestOutputs,'o',...
    'LineWidth',1,...
    'MarkerSize',6,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.3,0.9,0.2]);
    title(['Fuzzy Test - R =  ' num2str(1-gof.rmse)]);
    xlabel('Test Target');
    ylabel('Test Output');    
hold on
plot(population2,'r-','predobs');
    xlabel('Test Target');
    ylabel('Test Output');
 hold off
%% Errors
% Fuzzy Regression Train and Test Errors]
% Train
 fprintf('Fuzzy Regression Training "MSE" Is =  %0.4f.\n',fMSE)
 fprintf('Fuzzy Regression Training "RMSE" Is =  %0.4f.\n',fRMSE)
 fprintf('Fuzzy Regression Training "Mean Error" Is =  %0.4f.\n',ferror_mean)
 fprintf('Fuzzy Regression Training "STD Error" Is =  %0.4f.\n',ferror_std)
 fprintf('Fuzzy Regression Training "MAE" Is =  %0.4f.\n',mae(data.TrainTargets,fTrainOutputs))
% Test
 fprintf('Fuzzy Regression Testing "MSE" Is =  %0.4f.\n',fMSE1)
 fprintf('Fuzzy Regression Testing "RMSE" Is =  %0.4f.\n',fRMSE1)
 fprintf('Fuzzy Regression Testing "Mean Error" Is =  %0.4f.\n',ferror_mean1)
 fprintf('Fuzzy Regression Testing "STD Error" Is =  %0.4f.\n',ferror_std1)
 fprintf('Fuzzy Regression Testing "MAE" Is =  %0.4f.\n',mae(data.TestTargets,fTestOutputs))
% FireFly Regression Algorithm Train and Test Errors
% Train
 fprintf('FireFly Regression Training "MSE" Is =  %0.4f.\n',MSE)
 fprintf('FireFly Regression Training "RMSE" Is =  %0.4f.\n',RMSE)
 fprintf('FireFly Regression Training "Mean Error" Is =  %0.4f.\n',error_mean)
 fprintf('FireFly Regression Training "STD Error" Is =  %0.4f.\n',error_std)
 fprintf('FireFly Regression Training "MAE" Is =  %0.4f.\n',mae(data.TrainTargets,TrainOutputs))
% Test
 fprintf('FireFly Regression Testing "MSE" Is =  %0.4f.\n',MSE1)
 fprintf('FireFly Regression Testing "RMSE" Is =  %0.4f.\n',RMSE1)
 fprintf('FireFly Regression Testing "Mean Error" Is =  %0.4f.\n',error_mean1)
 fprintf('FireFly Regression Testing "STD Error" Is =  %0.4f.\n',error_std1)
 fprintf('FireFly Regression Testing "MAE" Is =  %0.4f.\n',mae(data.TestTargets,TestOutputs))
 
