character = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
% fparr = ["xl","lc2","yyj2","yyk2","jc","lzp","whj","swx","wh","ct","wjks2","lyx","dxh"];
% fparr = ["ct","wjks2"];
fparr = ["mlk","mlk2",];
% fparr = "znl2";
for i = 1 : length(fparr)
    fn = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
%     fn = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'];
%     fn = ['E','F','G','H','I','J','K'];
    basefp = sprintf("%s%s%s","D:\学习\研三\论文\data\",fparr(i),"\");
%     fn = ['A','B','C','D','E','F','G','H','I','J'];
%     fn = ['L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
    % fn = ['C','J','T','U','Y','Z'];
    % fn = ['R','S','T','U','V','W','X','Y','Z'];
    % fn = ['A','B','C'];
    % fn = 'C';
    dTNum = [];
    dNum = [];
    for i = 1 : length(fn)
        fp1 = sprintf("%s%s%s",basefp,fn(i),'_x.txt');
        fp2 = sprintf("%s%s%s",basefp,fn(i),'_y.txt');
%         fp1 = [basefp,fn(i),'_x.txt'];
%         fp2 = [basefp,fn(i),'_y.txt'];
        X = csvread(fp1);
        Y = csvread(fp2);
        processingData;
%        test2;
%        plotdata;
    end
    file_train_label = fopen('D:\学习\研三\论文\data\train_data_double\train_label','A');
    file_test_label = fopen('D:\学习\研三\论文\data\train_data_double\test_label','A');
    for i = 1 : length(fn)
        for j = 1 : dNum(i)
            fprintf(file_train_label,'%d\n',find(character == fn(i))-1);
        end
        for j = 1 : dTNum(i)
            fprintf(file_test_label,'%d\n',find(character == fn(i))-1);
        end
    end
    fclose(file_train_label);
    fclose(file_test_label);

end

% basefp = 'D:\学习\研三\论文\data\mlk\';
