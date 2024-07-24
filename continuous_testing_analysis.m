clear;
clc;

config = ReadYaml("config.yml");
TESTING_SUBSET = "severe";
EPOCH = 5;
PATH_TO_SUBSET = config.("subset_" + string(TESTING_SUBSET) + "_directory");
PATH_TO_SUBSET0_CONT_TESTING = config.("subset_0_continuous_testing_directory");
PATH_TO_SUBSET_CONT_TESTING = config.("subset_" + string(TESTING_SUBSET) + "_continuous_testing_directory");
NET_TYPE = string(config.net_type);
NET_IDENTIFIER = string(config.net_identifier);

% Standardize path to unix style (matlab handles it in Windows too):
PATH_TO_SUBSET = strrep(PATH_TO_SUBSET,'\\', '/');
PATH_TO_SUBSET = strrep(PATH_TO_SUBSET,'\', '/');
PATH_TO_SUBSET0_CONT_TESTING = strrep(PATH_TO_SUBSET0_CONT_TESTING,'\\', '/');
PATH_TO_SUBSET0_CONT_TESTING = strrep(PATH_TO_SUBSET0_CONT_TESTING,'\', '/');
PATH_TO_SUBSET_CONT_TESTING = strrep(PATH_TO_SUBSET_CONT_TESTING,'\\', '/');
PATH_TO_SUBSET_CONT_TESTING = strrep(PATH_TO_SUBSET_CONT_TESTING,'\', '/');
if ~exist(PATH_TO_SUBSET_CONT_TESTING, "dir")
    mkdir(PATH_TO_SUBSET_CONT_TESTING);
end

path0 = PATH_TO_SUBSET0_CONT_TESTING + '/' + "cont-test-results" +'/' + NET_TYPE + '/' + NET_IDENTIFIER + '/' + "epoch-" + string(EPOCH) + '/';
path = PATH_TO_SUBSET_CONT_TESTING + '/' + "cont-test-results" +'/' + NET_TYPE + '/' + NET_IDENTIFIER + '/' + "epoch-" + string(EPOCH) + '/';
%% 
NET_TYPE
NET_IDENTIFIER
pathToIds = PATH_TO_SUBSET + "/ids.csv";
ids = csvread(pathToIds);
if exist(path, "dir")   
    a=dir(path + '**/cont_test_signal_*.mat');
    if isempty(a)
        disp("Pulling files from dataset-all");
        a = dir(path0 + '**/cont_test_signal_*.mat');
    end
else
    disp("Pulling files from dataset-all");
    a=dir(path0 + '**/cont_test_signal_*.mat');
end
if length(a) < length(ids)
    disp("Warning: cont-test-results in the testing subset directory is incomplete ");
end   

a_filtered = struct("folder", cellstr(strings([length(ids),1])), ...
    "name", cellstr(strings([length(ids),1])), ...
    "id", num2cell(zeros([length(ids),1], "uint8")), ...
    "isValidation", num2cell(zeros([length(ids),1], 'logical')));

si = 1;
for i=1:length(a)
    s = strrep(a(i).name,"cont_test_signal_",'');
    sub_id = str2num(strrep(s,".mat",''));
    if ismember(sub_id, ids)
        a_filtered(si).folder = a(i).folder;
        a_filtered(si).name = a(i).name;
        a_filtered(si).id = sub_id;
        if contains(a(i).folder, "validation")
            a_filtered(si).isValidation = true;
        else
            a_filtered(si).isValidation = false;
        end
        si = si + 1;
    end    
    
end
%%
WINDOWS_SIZE_MIN=60;
LABEL = 0;
FREQ = 32;
N_THREADS = 4;

filt_sz=WINDOWS_SIZE_MIN*FREQ*60;

probability_threshold=0.2;

sub_info = zeros(length(ids), 2);
corr_proba_duration = zeros(length(ids), 2);
corr_proba_cli_events = zeros(length(ids), 2);


parpool(N_THREADS)
parfor_progress(length(ids));
parfor i=1:length(ids)
    cont_test_file = load(a_filtered(i).folder + "/" + a_filtered(i).name);
    sub_id = a_filtered(i).id;
    trainedSub = a_filtered(i).isValidation;
    %{
    cont_test_file = load(a(i).folder + "/" + a(i).name);
    
    if contains(a(i).folder, "validation")
        trainedSub = true;
    else
        trainedSub = false;
    end
    
    s = strrep(a(i).name,"cont_test_signal_",'');
    sub_id = str2num(strrep(s,".mat",''));
    if ~ismember(sub_id, ids)
        continue;
    else
        si = si + 1;
    end    
    %}

    LogicalStr = {'false', 'true'};
    fprintf("%d/%d, Subject: %d From train set: %s\n", i, length(ids), sub_id, ...
        LogicalStr{trainedSub+1});
    
    sub_info(i,:) = [sub_id trainedSub];
    
    labels = cont_test_file.labels;
    prediction_probabilities = cont_test_file.prediction_probabilities;
    predictions = cont_test_file.predictions;
    
    % Get only the relevant label:
    if LABEL < 5
        rel_labels = labels==LABEL;
    elseif LABEL == 123
        rel_labels = labels==1 | labels==2 | labels==3;
    end
    
    % tmp1 is binary series, giving the the normalized total duration of
    % the relevant events inside a rolling window preceeding a given
    % number.
    tmp1=filter(ones(1,filt_sz)/filt_sz,1,rel_labels);
    
    % tmp1b is also a binary series, giving the number of relevant clinical
    % events inside a rolling window preceeding a given moment.
    tmp1b = zeros(1, length(rel_labels), 'logical');
    
    % Rolling window determining clinical events in it:
    for j=filt_sz:length(rel_labels)
        % Get the window:
        window = rel_labels(j-filt_sz+1:j);
        clinical_events = 0;
        k = 1;
        
        % Find start of events:
        while k < filt_sz
            if window(k) > 0
                d = 1;
                % Find how long event lasts:
                while k+d<=filt_sz && window(k + d) > 0
                    d = d + 1;
                end
                % if event lasts 10s, count it towards clinical events
                if d >= FREQ * 10
                    clinical_events = clinical_events + 1;
                end
                % move one step after the event to search for others:
                k = k + d;
            else
                % no event starts at k. Moving forward.
                k = k + 1;
            end
        end
        tmp1b(1,j) = clinical_events;
    end
    
    % tmp2=filter(ones(1,filt_sz)/filt_sz,1,prediction_probabilities(:,LABEL+1)>=probability_threshold);
    if LABEL < 5
        tmp2=filter(ones(1,filt_sz)/filt_sz,1,prediction_probabilities(:,LABEL+1));
    elseif LABEL == 123
        tmp2=filter(ones(1,filt_sz)/filt_sz,1, prediction_probabilities(:,2) ...
            + prediction_probabilities(:,3) + prediction_probabilities(:,4));    
    end
    
    
    [R1,PValue1] = corr([tmp1' tmp2]);
    fprintf("%d/%d, Subject: %d Probability and duration corr (pvalue): %.4f (%.4f)\n", ...
        i, length(ids), sub_id, R1(1,2), PValue1(1,2));

    corr_proba_duration(i,:) = [R1(1,2) PValue1(1,2)];
    
    [R2,PValue2] = corr([tmp1b' tmp2]);
    fprintf("%d/%d, Subject: %d Probability and clinical events corr (pvalue): %.4f (%.4f)\n", ...
        i, length(ids), sub_id, R2(1,2), PValue2(1,2));
    corr_proba_cli_events(i,:) = [R2(1,2) PValue2(1,2)];
    
    % figure;
    % title("Subject: " + string(sub_id))
    % corrplot([tmp1' tmp2])
    % pause
    parfor_progress;
end
parfor_progress(0);
delete(gcp('nocreate'))
%%
mean(corr_proba_duration(:,2))
mean(corr_proba_cli_events(:,2))

varNames = {'sub_id', 'trained', 'corr_proba_duration', 'pval_proba_duration', ...
    'corr_proba_cli_events', 'pval_proba_cli_events'};
resultsTable = table(sub_info(:,1), sub_info(:,2), corr_proba_duration(:,1), corr_proba_duration(:,2), ...
    corr_proba_cli_events(:,1), corr_proba_cli_events(:,2), 'VariableNames', varNames);

%%
mkdir(path);
hash = string(mlreportgen.utils.hash(NET_IDENTIFIER));
% Excel has a total max path length of ~ 218 chars
if strlength(path) < (218-15-33)  % 15 chars for corr_win_<winSize>.xlsx and 33 for _<hash>
    table_path = path + "corr_win" + string(WINDOWS_SIZE_MIN) + "_" + hash + ".xlsx";
else
    table_path = path + "corr_win" + string(WINDOWS_SIZE_MIN) + ".xlsx";
end    
writetable(resultsTable, table_path, "Sheet", "Label-"+string(LABEL));