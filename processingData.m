data1 = X;
data2 = Y;

%���ò���Ƶ��
fs=90; 
fss=fs/2;
ts=1/fs; 

 L=length(data1);
 t=0:ts:(L-1)/fs;

%butter�˲�x
[bb, aa] = butter(3,[0.1/fss,8/fss]);
linVelHP1 = filtfilt(bb, aa, data1);
linVelHP2 = filtfilt(bb, aa, data2);
[dd,cc] = butter(3,[0.1/fss,10/fss]);
ft1 = filtfilt(dd, cc, data1);
ft2 = filtfilt(dd, cc, data2);
dft1 = decimate(ft1,2);
dft2 = decimate(ft2,2);
%%

%%
%��ͼ
% figure;

%��һ�׵���Ĳ���ͼ
diffA1 = diff(linVelHP1);
LA = length(diffA1);
tA = 0:ts:(LA-1)/fs;
subplot(2,1,1);
plot(tA,diffA1);
set(gca,'linewidth',1);
% xlim([0.5 15.5]) 
xlabel('Time(s)')
ylabel('velocity')
axis([-inf inf -4 4]);

diffA2 = diff(linVelHP2);
LA = length(diffA2);
tA = 0:ts:(LA-1)/fs;
subplot(2,1,2);
plot(tA,diffA2);
set(gca,'linewidth',1);
% %xlim([0.5 15.5]) 
xlabel('Time(s)')
ylabel('velocity')
axis([-inf inf -4 4]);

fid = fopen('D:\ѧϰ\����\����\data\train_data_double\train_data','A');
fidDoubleTest = fopen('D:\ѧϰ\����\����\data\train_data_double\test_data','A');
% fidSingleTest = fopen('D:\ѧϰ\����\����\data\train_data_single\test_data','A');
%%

%%
%�ָ�
formatSpec = '%.2f,';
[front1,tail1,energy1] = division(diffA1,0.1);
[front2,tail2,energy2] = division(diffA2,0.05);
front = [];
tail = [];
i = 1;
j = 1;
% while i <= length(front2)
%     front = [front,front2(i)];
%     tail = [tail,tail2(i)];
%     i=i+1;
% end
while i <= length(front1) && j <= length(front2)
%     fprintf("%d,%d\n",i,j);
    if tail1(i)>=front2(j) && front1(i)<=tail2(j)
        if front1(i)<front2(j)
            front = [front,front2(j)];
        else
            front = [front,front1(i)];
        end
        if tail1(i)<tail2(j)
            tail = [tail,tail1(i)];
        else
            tail = [tail,tail2(j)];
        end
        if tail(end)-front(end) < 30 || tail(end)-front(end) > 225
            front = front(1:end-1);
            tail = tail(1:end-1);
        end
    elseif tail1(i)<front2(j)
        j = j-1;
    elseif front1(i)>tail2(j)
        i = i-1;
    end
    i = i+1;
    j = j+1;
end

%д�ļ�
% doubleTestNum = ceil((length(front)-2)/10);
% doubleTestNum = 0;
% doubleTestNum��ʾ����Ϊ���Լ�����������
doubleTestNum = length(front)-1;
dTNum = [dTNum,doubleTestNum];
dNum = [dNum,length(front)-1-doubleTestNum];
% ��Ҫ��һ��
for i = 2 : length(front)
    x = 1:(tail(i)-front(i)+1);
    xx = linspace(1,(tail(i)-front(i)+1),90);
    yy1 = spline(x,ft1(front(i):tail(i)),xx);
    yy2 = spline(x,ft2(front(i):tail(i)),xx);
    ma1 = max(yy1);
    mi1 = min(yy1);
    ma2 = max(yy2);
    mi2 = min(yy2);
    yy1 = (yy1-mi1)./(ma1-mi1);
    yy2 = (yy2-mi2)./(ma2-mi2);
%     yy1 = ft1(front(i):tail(i));
%     yy2 = ft2(front(i):tail(i));
    yy = [yy1,yy2];
    if i <= 1+doubleTestNum
        fprintf(fidDoubleTest,formatSpec,yy);
        fprintf(fidDoubleTest,"\n");
    else
        fprintf(fid,formatSpec,yy);
        fprintf(fid,"\n");
    end
end
fclose(fid);
fclose(fidDoubleTest);

function [front,tail,energy] = division(diffA,threshold)
    LA = length(diffA);
    front = [];
    tail = [];
    %����С���ϲ��ķ�ʽ���в�����ȡ��������ƽ��������0.01-0.025���䣬����С��������������threshold����Ϊ�����Ʋ���
    windowLen = 18;  %С����ȣ�7���㣬0.05s
    energy = [];
    for i = 1 : LA/windowLen
        energy = [energy,mean(sum(diffA((i-1)*windowLen+1:i*windowLen).*diffA((i-1)*windowLen+1:i*windowLen),2))];
    end
        energy = [energy,mean(sum(diffA(floor(LA/windowLen)*windowLen+1:end).*diffA(floor(LA/windowLen)*windowLen+1:end),2))];
    count = 0;
    maxCount = 4; %����������ֵ������С��������5��Ϊ0.5s
    for i = 1 : length(energy)
        if energy(i) > threshold
            count = count+1;
        elseif count >= maxCount
%             front = [front,(i-1-count)*windowLen+floor(windowLen/4)];
            front = [front,(i-1-count)*windowLen];
%             tail = [tail,(i-1)*windowLen-floor(windowLen/4)];
            tail = [tail,(i-1)*windowLen];
            count = 0;
        else
            count = 0;
        end
    end
end
%     ��һ�׵���Ĳ���ͼ
%     ���ֵ�������غ�����ƫ�ȡ���ȡ�Ƶ��ϵ����С��ϵ��
%     ma1 = max(yy1);
%     mi1 = min(yy1);
%     me1 = mean(yy1);
%     pk1 = ma1-mi1;
%     av1 = mean(abs(yy1));
%     va1 = var(yy1);
%     st1 = std(yy1);
%     ku1 = kurtosis(yy1);
%     sk1 = skewness(yy1);
%     rm1 = rms(yy1);
%     S1 = rm1/av1;
%     C1 = pk1/rm1;
%     I1 = pk1/av1;
%     xr1 = mean(sqrt(abs(yy1)))^2;
%     L1 = pk1/xr1;			%ԣ������
% 
%     ma2 = max(yy2); 			%���ֵ
%     mi2 = min(yy2); 			%��Сֵ	
%     me2 = mean(yy2); 			%ƽ��ֵ
%     pk2 = ma2-mi2;			%��-��ֵ
%     av2 = mean(abs(yy2));		%����ֵ��ƽ��ֵ(����ƽ��ֵ)
%     va2 = var(yy2);			%����
%     st2 = std(yy2);			%��׼��
%     ku2 = kurtosis(yy2);		%�Ͷ�
%     sk2 = skewness(yy2);               %ƫ��
%     rm2 = rms(yy2);			%������
%     S2 = rm2/av2;			%��������
%     C2 = pk2/rm2;			%��ֵ����
%     I2= pk2/av2;			%��������
%     xr2 = mean(sqrt(abs(yy2)))^2;
%     L2 = pk2/xr2;			%ԣ������
%     yy_corrcoef = corrcoef(yy1,yy2);
%     yy = [yy1,yy2,ma1,mi1,me1,pk1,av1,va1,st1,ku1,sk1,rm1,S1,C1,I1,xr1,L1,ma2,mi2,me2,pk2,av2,va2,st2,ku2,sk2,rm2,S2,C2,I2,xr2,L2,yy_corrcoef(1)];
%     yy_corrcoef = corrcoef(yy1,yy2);

