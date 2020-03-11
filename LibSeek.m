
% Author: Bo Li <boli@swin.edu.au>
% Created: 03/10/2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Algorithm Overview  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                 %
% 1. LibSeek employs both explicit and implicit information when recommending useful libraries for%
%    given apps. Besides, it employs the neighborhood information between similar apps and similar%
%    libraries, respectively, to enhance the diversity of the recommendation.                     %
%                                                                                                 %
% 2. This program provides a prototype of LibSeek, which can be executed with Mathlab 2017b or    %
%    later version.                                                                               %
%                                                                                                 %
% 3. You are recommended to use the MALib as the dataset, which can be find from Github           %
%    (https://github.com/malibdata/MALib-Dataset)                                                 %
%                                                                                                 %   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% STEP 0: INITIALIZE PARAMETERS  %%%

	% Bellow are some parameters.
	% You can setup them by yourself or refer to paper: 
	%    Diversified Third-Party Library Prediction for Mobile App Development (Qiang He et al.)
	%    publised on IEEE Transactions on Software Engineering

	weight = 17; %weight of the explicit information
	lambda = 0.01; %regularization for training
	iteration = 15; %times of iteration, most of the time, it is less than 30.
	factor = 20;  %The number of latent factors, i.e., the degree of dimensionality in U and I
	alpha = 0.5; %how much the MF will consider the neiborhood similarity.
	topk = 10;  %number of neighbors.
	listLength = 10; % the length of each recommendation list, i.e., how many items in it.
	kickout = 3; % defines the scale of the testset. For example, when kickout=3, each app will be deleted 3 used libraries to create the testset.
	left = 5; % for each app, the minimum number of libraries left in it when creating the training set.
	minOfLibrary = 10;  % for each app, when the number of libraries it used is less then minOfLibrary, it will be excluded from the dataset.
	minOfApp = 6; % for each library, when the number of apps that used itself is less then minOfApp, it will be excluded from the dataset.


%%% STEP 1: READ THE DATASET FROM FILE  %%%

	% !!! This step can be skipped if its not the first time to use it.
	%Read the dataset (MALib), put the relation.csv to the current folder of Mathlab.
	tmp = csvread('relation.csv'); % read the original dataset
	tmp_size = size(tmp,1); 
	size_apk=max(tmp(:,1));  % the number of apps
	size_lib=max(tmp(:,2));  % the number of libraries

	%create a matrix, i.e., convert the original relation.csv to matrix, as Matlab can only use matrix.

	relation = zeros(size_apk,size_lib);  % the relation matrix;
	for i=1:size(tmp,1)
		relation(tmp(i,1),tmp(i,2))=1;
	end

	sum_relation = sum(relation);
	size_lib = size(sum_relation,2);

%%% STEP 2: DATA CLEAN.   %%%

	% !!! This step can be skipped if its not the first time to use it.
	%Remove some data that is not useful in the experiment according to  minOfLibrary and minOfApp %%%%%%%%%%%%%%%%%%%%%%%
	tic;   
		size_relation = size(relation);
		size_new = [2,2];
		count = 0;
		while (size_relation(1)~= size_new(1) || size_relation(2)~=size_new(2)) && size_new(1)>1 && size_new(2)>1
			size_relation = size_new;
			rowCnt = sum(relation,2);
			relation(find(rowCnt<minOfLibrary),:) = [];

			colCnt = sum(relation);
			relation(:,find(colCnt<minOfApp)) = [];
			
			size_new = size(relation);
			count = count+1;
			display(strcat(' |-->Current rows:',num2str(size_new(1)),' columns:',num2str(size_new(2)),' Interate:',num2str(count)));
		end    
		display(strcat(' |-->We have deleted rows:',num2str(size_apk - size_new(1) ),' columns:',num2str(size_lib - size_new(2))));
		
		csvwrite('relation_new.csv',relation);
		size_apk = size(relation,1);
		size_lib = size(relation,2);
		display(strcat(' |--> the new data set has been saved to relation_new.csv'));  
	t = toc;
	display(strcat(' |--> Time consuption  --> ',num2str(t),' Seconds',10));  


%%% STEP 3: SPLIT DATASET TO TESTSET AND TRAININGSET.    %%%

	% reload the dataset if you have skipped the above two steps.
	relation = csvread("relation_new.csv");
	Remain = relation;
	size_apk = size(relation,1);
	size_lib = size(relation,2);
	Remove = zeros(size_apk,size_lib);  % used for storing the testset
	Unused = zeros(size_apk,1);         % used for storing the trainingset

	tic;
	if kickout > 0  % kickout stands for the number of libraies removed from the dataset, i.e., the removed libraries form the testset.
			
		for i=1:size_apk
			itemU = Remain(i,:); 
			uSer = find(itemU==1);% find all libraies, i.e., the non-zero elements, of the ith app.
			
			j = 0;
			while j<kickout
				if size(uSer,2)  > left
					r = ceil(rand()*size(uSer,2));
					Remain(i,uSer(r)) = 0; %  randomly remove the corresponding library;
					itemU(uSer(r)) = 0;
					Remove(i,uSer(r)) = 1;
					uSer = find(itemU==1);
					j = j+1;
				else
					Unused(i) = 1;
					break;
				end
			end
		end
		trainSet = sum(sum(Remain)); % how many elements (usage information) in the trainingset
		testSet = sum(sum(Remove));  % how many elements in the testset.
		failure = sum(Unused); % how many apps that can not be splitted into trainingset and testset.
		SUM = sum(Remain);
		None = size(find(SUM==0),2);  %% variable None is used to calculate the number of libraries that haven't been used after splitting.
		display(strcat('O--- finished, trainingset: <',num2str(trainSet),'> testSet： <',num2str(testSet),'> failures in app： <',num2str(failure),'>',' and library： <',num2str(None),'>',10));
		if None>0 || failure>0                
			%Sound - birds, a reminder when failure appears.
			load chirp
			sound(y,Fs) 
		end
		% save the trainset and testset to .csv files.
		csvwrite(strcat('libseek_testSet_',num2str(kickout),'.csv'),Remove);
		csvwrite(strcat('libseek_trainSet_',num2str(kickout),'.csv'),Remain);
	else
		display(strcat("Warning: Please setup the parameter kickout!!!",10));
	end
	t = toc;
	display(strcat(' |--> Time consumption for data splitting  --> ',num2str(t),' Seconds',10));  



%%% STEP 4: RUN THE EXPERIMENT.  %%%
	
	% This is the most important step. In this step, LibSeek first imports the testset and the training set, then computes the similarities
    % and creates the neiborhood information. Finally, it combines the MF algorithm with the neiborhood information to give out the recommendations.

	%-----------------------------------------
	% sub step (1) compute the similarity 
	%-----------------------------------------
	% read the training set
	Remain = csvread(strcat('libseek_trainSet_',num2str(kickout),'.csv'));
	% compute the similarity between libraries and apps, includding maxPI, maxPU, maxVI, maxVU        
	size_lib = size(Remain,2);
	size_apk = size(Remain,1);

	tic;
	relation = Remain;
	sum_relation = sum(Remain);
	ref_relation = Remain';
	sum_ref_relation = sum(ref_relation);
	
	% app similarity
	t = toc;
	display(strcat('  |--  Init time：',num2str(t),'Seconds'));
	
	display('  |----------------------------------------------------<');
	tic;
	simiU = zeros(size_apk,size_apk);
	for u=1:size_apk
		user_u = ref_relation(:,u);
		fz_tmp = relation*user_u;
		fm_tmp = (sum_ref_relation(u) + sum_ref_relation)' - fz_tmp;
		simiU(:,u) = fz_tmp./fm_tmp;
		simiU(u,u) = 0;%the similar of itself is 0.
	end
	[sortA,xiabiao]=sort(simiU,'descend'); 
	maxVU = zeros(topk,size_apk);
	maxPU = zeros(topk,size_apk);
	maxVU = sortA(1:topk,:);  
	maxPU = xiabiao(1:topk,:);  
	%normalization
	maxW = sum(maxVU);
	for u=1:size_apk
		maxVU(:,u) = maxVU(:,u)/maxW(u);
	end
	% save to file
	csvwrite(strcat('similarity/maxVU_',num2str(kickout),'.csv'),maxVU);
	csvwrite(strcat('similarity/maxPU_',num2str(kickout),'.csv'),maxPU);
	
	t = toc;
	display(strcat('  |--  APP similarity computing time： ',num2str(t),' s'));
	display('  |----------------------------------------------------<');
	
	tic;
	% lib similarity
	simiI = zeros(size_lib,size_lib);
	for i=1:size_lib
		item_i = relation(:,i);  
		fenzi_tmp = ref_relation*item_i;
		fenmu_tmp = (sum_relation(i) + sum_relation)' - fenzi_tmp;
		simiI(:,i) = fenzi_tmp./fenmu_tmp;
		simiI(i,i)=0;%similarity of itself is 0
	end
	
	csvwrite(strcat('similarity/simiI_',num2str(kickout),'.csv'),simiI);	% similarity between libraries. used for calculating MILD later.
	
	[sortA,xiabiao]=sort(simiI,'descend');  
	maxVI = zeros(topk,size_lib);
	maxPI = zeros(topk,size_lib);
	maxVI = sortA(1:topk,:);  
	maxPI = xiabiao(1:topk,:); 
	%normalization
	maxW = sum(maxVI);
	for i=1:size_lib
		maxVI(:,i) = maxVI(:,i)/maxW(i);
	end
	
	csvwrite(strcat('similarity/maxPI_',num2str(kickout),'.csv'),maxPI);
	csvwrite(strcat('similarity/maxVI_',num2str(kickout),'.csv'),maxVI);
	
	t = toc;
	display(strcat('  |--  Library similarity computing time： ',num2str(t),'s',10));
	
	
	%-----------------------------------------
	% sub step (2) perform the recommendation
	%-----------------------------------------
			
	logweight = zeros(1,size_lib);
	logweight = weight./(log(sum(Remain)+1)+1);
	% read the similarity between libraries and apps, includding maxPI, maxPU, maxVI, maxVU
	% V means the value of similarity while P means the position or id of each library or app, respectively.
	PI = csvread(strcat('similarity//maxPI_',num2str(kickout),'.csv'));
	PU = csvread(strcat('similarity//maxPU_',num2str(kickout),'.csv'));
	VI = csvread(strcat('similarity//maxVI_',num2str(kickout),'.csv'));
	VU = csvread(strcat('similarity//maxVU_',num2str(kickout),'.csv'));

	maxPI = PI(1:topk,:); % only select their top k neighbors
	maxPU = PU(1:topk,:);
	maxVI = VI(1:topk,:);
	maxVU = VU(1:topk,:);

	size_lib = size(Remain,2);
	size_apk = size(Remain,1);

	Position = zeros(size_apk,listLength);

	C = zeros(size_apk,size_lib);
	rng(11); %random seed, can be set to any value. Comment it can lead to a slight vary of results between different attempts.
	X = rand(size_apk,factor)+0.01;
	rng(16);  %random seed.
	Y = rand(size_lib,factor)+0.01;
	for i=1:size_lib
		C(:,i) = 1 + logweight(i)* Remain(:,i);
	end
	P = Remain;
	% please refer to the paper to find the meaning of those variables
	YtY = zeros(factor,factor);
	Cu = zeros(size_lib,1);
	Pu = zeros(size_lib,1);

	XtX = zeros(factor,factor);
	Ci = zeros(size_apk,1);
	Pi = zeros(size_apk,1);
	Ii = ones(size_apk,1); 

	Xu = zeros(factor,1);  
	Yi = zeros(factor,1); 
	
	time_consume = 0;

	for epochs = 1:iteration
		
		display(strcat(10,'  O----  The <',num2str(epochs),'>th iteration:'));
		t1=0;
		tic;
		YtY = Y'*Y;
		t1 = toc;
		display(strcat('    |--> YtY computation time consumption--> ',num2str(t1),' s'));
		t2=0;
		tic;
		for u = 1:size_apk
			Cu = C(u,:)'; 
			Pu = P(u,:)';
			hou = Cu.*Pu;
			hou = Y'*hou;
			Nu = X(maxPU(:,u),:)'; 
			WuNormal = maxVU(:,u); % Normalized all similarities of neighborhood
			Wu = WuNormal./sum(WuNormal);
			hou = hou + alpha*Nu*Wu;
			Cu = Cu-1;
			qian = Y';
			for j=1:size_lib
				qian(:,j) = qian(:,j)*Cu(j);
			end
			qian = qian*Y;
			qian = qian+YtY;
			qian = qian+lambda+alpha;  
			Xu = (qian)^-1 * hou;
			X(u,:) = Xu';
		end
		t2 = toc;
		display(strcat('    |--> X computing time  --> ',num2str(t2),' s'));

		t3=0;
		tic;
		XtX = X'*X;
		t3 = toc;
		display(strcat('    |--> XtX computing time  --> ',num2str(t3),' s'));
		
		t4=0;
		tic;
		for i = 1:size_lib
			% ( X^t*(Ci-I)*X + XtX + lam_I)^(-1)*X^t*Ci*P(:,i)
			Ci = C(:,i);
			Pi = P(:,i);
			hou = Ci .* Pi;
			hou = X'*hou;
			Ni = Y(maxPI(:,i),:)'; 
			WiNormal = maxVI(:,i);
			Wi = WiNormal./sum(WiNormal); % normalized similarities
			hou = hou + alpha*Ni*Wi;                    
			Ci = Ci-1; 
			qian = X';
			for j=1:size_apk
				qian(:,j) = qian(:,j)*Ci(j);
			end
			qian = qian*X;
			qian = qian + XtX;
			qian = qian + lambda + alpha;           % normalized similarities
			Yi = (qian)^-1 * hou;
			Y(i,:) = Yi';
		end
		t4 = toc;
		display(strcat('    |--> Y computing time  --> ',num2str(t4),' s'));
		display(strcat('    |==> Total iteration Time  --> ',num2str(t1+t2+t3+t4),' s'));
		
	end  % end of iteration

	tic;
	Prediction = X*Y';  %prediction matrix, each elements has a value that stands for the probability of one corresponding libraries is interesting for the corresponding app.
	t = toc;
	display(strcat('    |--> Prediction time --> ',num2str(t)));

	tic;
	for u=1:size_apk
		pre_u = Prediction(u,:);
		pre_u(find(Remain(u,:)==1))=0; %remove the existing libraries
		[sortA,xiabiao]=sort(pre_u,'descend' );%sort them according to the values in prediction matrix.
		Position(u,1:listLength)=xiabiao(1:listLength);        
	end
	% write the predition result to csv file
	% In this way, we can analyze the result via different metrics, such as MAP, MRR, etc.
	csvwrite(strcat('result_',num2str(kickout),'_Prediction.csv'),Position);
	t = toc;
	display(strcat('    |--> Write to file  --> ',num2str(t)));



%%% SETP 5: COLLECT RESULT  %%%

    tic;
	% for each algorithm, read the prediction files and compute the result.        
	display(' O-------------------------------------------------------------------------O');
	simiI = csvread(strcat('similarity//simiI_',num2str(kickout),'.csv'));
	
	Prediction = csvread(strcat('result_',num2str(kickout),'_Prediction.csv'));
	% testSet
	RESERVE = csvread(strcat('libseek_testSet_',num2str(kickout),'.csv'));
	
	MAP = zeros(size_apk,1);  
	MP = zeros(size_apk,1); 
	MR = zeros(size_apk,1); 
	Cover = 0; %coverage 
	MILD = zeros(size_apk,1);
	
	t = toc;
	display(strcat('    |--> Initial time  --> ',num2str(t)));
	
	tic;
	% size(find(recommendation>0),2) is used to compute the Cover matric
	recommendation = zeros(1,size_lib);
	% temp matrix
	temp = zeros(size_apk,listLength);
	tmp = zeros(listLength,listLength);
	
	
	for u=1:size_apk
		% get the prediction result and removed libraries;
		reserve = RESERVE(u,:);
		Position=Prediction(u,1:listLength);

		ap = 0; %MAP
		find_count = 0;
		if size(Position,1)>=1 && size(Position,2)>=1
			ReallistLength=size(Position,2);
		else
			ReallistLength=0;
		end
		for i=1:ReallistLength
			recommendation(Position(i)) = recommendation(Position(i)) + 1;
			if reserve(Position(i))==1
				find_count = find_count + 1;
				ap = ap + find_count/i;
			end
		end
		if find_count > 0
			MAP(u) = ap/find_count;
			MP(u) = find_count/listLength;
			MR(u) = find_count/kickout;
		end
		
		% inter list diversity, MILD
		if ReallistLength  % to confirm the length of the predictin list is larger than 0.
			temp = simiI(:,Position);
			tmp = temp(Position,:);
			MILD(u) = sum(sum(tmp))/(listLength-1)/listLength;
		end
	end
	t = toc;
	display(strcat('    |--> Accuracy computing time  --> ',num2str(t)));
	
	tic;
	% coverage
	Cover = size(find(recommendation>0),2)/size_lib;
	
	
	t = toc;
	display(strcat('    |--> Diversity computing time  --> ',num2str(t)));
	
	display('    O==> Final Result==============');
	display(strcat('    |--> MAP    ->',num2str(mean(MAP)),' '));
	display(strcat('    |--> MP    ->',num2str(mean(Pre)),' '));
	display(strcat('    |--> MR     ->',num2str(mean(MR)),' '));
	display(strcat('    |--> Cover  ->',num2str(Cover),' '));
	display(strcat('    |--> MILD    ->',num2str(mean(MILD)),' '));
