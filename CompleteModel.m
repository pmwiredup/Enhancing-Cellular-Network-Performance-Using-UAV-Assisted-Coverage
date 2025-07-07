clear; 
close all; 
clc;

% n/w param
R = 1000; % Cell circumradius(m)
N_cells = 19;% No of hexagonal cells
lambda_users = 50;% Avg user density per cell (users per km^2)
P_tx_BS = 46;% BS P_t in dBm 
P_tx_UAV = 23; % UAV P_t in dBm
noise_power = -174;% Noise PSD in dBm/Hz
bandwidth = 10e6;% System bandwidth in Hz
path_loss_exp = 4; % Path loss exp
freq_reuse_factor = 7; % Freq reuse factor
SINR_threshold = 10; % SINR threshold in dB for BS/UAV selection

%UAV Parameters 
UAV_height_min = 10;  % Min UAV height (m) 
UAV_height_max = 120; % Max UAV height (m) 
UAV_power_max = 23;% Max UAV power (dBm)
fc_GHz = 2;% Carrier freq in GHz

% User Distribution Parameters for Non-Uniform Clustering
cluster_types = {'hotspot', 'commercial', 'residential', 'sparse'};
cluster_probabilities = [0.2, 0.3, 0.4, 0.1]; % Probability of each cluster type
hotspot_intensity = 3.0; % Multiplier for hotspot areas
commercial_intensity = 2.0;% Multiplier for commercial areas
residential_intensity = 1.2;% Multiplier for residential areas
sparse_intensity = 0.3;% Multiplier for sparse areas

%Generate Hexagonal Cell Structure
ISD = sqrt(3) * R;
cell_centers = generate_hex_centers_rotated(R);

% Plot network structure
figure(1);
plot_hexagonal_cells_rotated(cell_centers, R);
title('19-Cell Hexagonal Network with UAV Deployment and Clustered Users');
xlabel('Distance (m)'); ylabel('Distance (m)');
grid on; axis equal;

% Deploy Base Stations
BS_positions = cell_centers;
N_BS = size(BS_positions, 1);

hold on;
plot(BS_positions(:,1), BS_positions(:,2), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

%Generate Users with Non-Uniform Clustering
fprintf('Generating users with non-uniform clustering patterns...\n');
all_users = [];
user_cell_assignment = [];
user_cluster_types = {};

for cell_idx = 1:N_cells
    cell_area = 3 * sqrt(3) * R^2 / 2;
    cell_area_km2 = cell_area / 1e6;
    
    % Determine cluster type for this cell
    cluster_type = cluster_types{randsample(length(cluster_types), 1, true, cluster_probabilities)};
    
    % Adjust user density based on cluster type
    switch cluster_type
        case 'hotspot'
            effective_lambda = lambda_users * hotspot_intensity;
        case 'commercial'
            effective_lambda = lambda_users * commercial_intensity;
        case 'residential'
            effective_lambda = lambda_users * residential_intensity;
        case 'sparse'
            effective_lambda = lambda_users * sparse_intensity;
    end
    
    N_users_cell = poissrnd(effective_lambda * cell_area_km2);
    
    %Generate clustered user positions
    users_in_cell = generate_clustered_users_in_hex(cell_centers(cell_idx, :), R, N_users_cell, cluster_type);
    
    % Store users
    all_users = [all_users; users_in_cell];
    user_cell_assignment = [user_cell_assignment; cell_idx * ones(N_users_cell, 1)];
    user_cluster_types = [user_cluster_types; repmat({cluster_type}, N_users_cell, 1)];
    
    fprintf('Cell %d: %s cluster with %d users\n', cell_idx, cluster_type, N_users_cell);
end

N_total_users = size(all_users, 1);

% Plot users based on cluster type
plot_clustered_users(all_users, user_cluster_types);

%Optimize UAV Positions
fprintf('Optimizing UAV positions for %d cells...\n', N_cells);
UAV_positions = optimize_UAV_positions_enhanced(cell_centers, all_users, user_cell_assignment, R, user_cluster_types);

% Plot optimized UAV positions
plot(UAV_positions(:,1), UAV_positions(:,2), 'y^', 'MarkerSize', 12, 'MarkerFaceColor', 'y');
legend('Cell Boundaries', 'Base Stations', 'Hotspot Users', 'Commercial Users', ...
       'Residential Users', 'Sparse Users', 'UAVs', 'Location', 'best');

% Calculate SINR and Throughput for All Users
fprintf('Calculating SINR and throughput for all users...\n');

% Initialize arrays
SINR_BS = zeros(N_total_users, 1);
SINR_UAV = zeros(N_total_users, 1);
throughput_BS = zeros(N_total_users, 1);
throughput_UAV = zeros(N_total_users, 1);
user_assignment = zeros(N_total_users, 1); % 0: BS, 1: UAV
user_distances_BS = zeros(N_total_users, 1);
user_distances_UAV = zeros(N_total_users, 1);

for user_idx = 1:N_total_users
    user_pos = all_users(user_idx, :);
    serving_cell = user_cell_assignment(user_idx);
    serving_BS = BS_positions(serving_cell, :);
    serving_UAV = UAV_positions(serving_cell, :);
    
    %distance to BS and UAV
    d_BS = norm(user_pos - serving_BS);
    d_UAV_2D = norm(user_pos - serving_UAV(1:2));
    d_UAV_3D = sqrt(d_UAV_2D^2 + serving_UAV(3)^2);
    
    user_distances_BS(user_idx) = d_BS;
    user_distances_UAV(user_idx) = d_UAV_3D;
    
    % SINR from BS
    SINR_BS(user_idx) = calculate_SINR_BS(user_pos, serving_BS, BS_positions, ...
                                         user_cell_assignment, P_tx_BS, path_loss_exp);
    
    %SINR from UAV
    SINR_UAV(user_idx) = calculate_SINR_UAV(user_pos, serving_UAV, UAV_positions, ...
                                           user_cell_assignment, P_tx_UAV, fc_GHz);
    
    % throughput
    throughput_BS(user_idx) = bandwidth * log2(1 + 10^(SINR_BS(user_idx)/10)) / 1e6;
    throughput_UAV(user_idx) = bandwidth * log2(1 + 10^(SINR_UAV(user_idx)/10)) / 1e6;
    
    % User assignment based on threshold
    if SINR_BS(user_idx) >= SINR_threshold
        user_assignment(user_idx) = 0; % Served by BS
    else
        user_assignment(user_idx) = 1; % Served by UAV
    end
end

%Performance Analysis
users_served_by_BS = sum(user_assignment == 0);
users_served_by_UAV = sum(user_assignment == 1);
coverage_improvement = users_served_by_UAV / N_total_users * 100;

fprintf('\n=== UAV-Assisted Network Performance with Clustered Users ===\n');
fprintf('Total users: %d\n', N_total_users);
fprintf('Users served by BS: %d (%.1f%%)\n', users_served_by_BS, users_served_by_BS/N_total_users*100);
fprintf('Users served by UAV: %d (%.1f%%)\n', users_served_by_UAV, users_served_by_UAV/N_total_users*100);
fprintf('Coverage improvement: %.1f%%\n', coverage_improvement);
fprintf('Average BS SINR: %.2f dB\n', mean(SINR_BS));
fprintf('Average UAV SINR: %.2f dB\n', mean(SINR_UAV));
fprintf('Average BS throughput: %.2f Mbps\n', mean(throughput_BS));
fprintf('Average UAV throughput: %.2f Mbps\n', mean(throughput_UAV));

% Analyze performance by cluster type
analyze_performance_by_cluster(user_cluster_types, SINR_BS, SINR_UAV, throughput_BS, throughput_UAV, user_assignment);

% Plot 2: Cluster Distribution Analysis
figure(2);

subplot(3,1,1);
histogram(SINR_BS, 30, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
xlabel('SINR (dB)'); ylabel('Number of Users');
title('BS SINR Distribution (Clustered Users)');
xline(SINR_threshold, 'r--', 'LineWidth', 2, 'Label', '10 dB Threshold');
grid on;

subplot(3,1,2);
histogram(SINR_UAV, 30, 'FaceColor', 'green', 'FaceAlpha', 0.7);
xlabel('SINR (dB)'); ylabel('Number of Users');
title('UAV SINR Distribution (Clustered Users)');
xline(SINR_threshold, 'r--', 'LineWidth', 2, 'Label', '10 dB Threshold');
grid on;

subplot(3,1,3);
scatter(user_distances_BS, SINR_BS, 50, categorical(user_cluster_types), 'filled');
colormap(lines(4));
colorbar('Ticks', 1:4, 'TickLabels', cluster_types);
xlabel('Distance to BS (m)'); ylabel('SINR (dB)');
title('SINR vs Distance by Cluster Type');
grid on;

%Performance Comparison by Cluster Type
figure(3);
plot_cluster_performance_comparison(user_cluster_types, SINR_BS, SINR_UAV, throughput_BS, throughput_UAV, user_assignment);

%UAV Optimization Results
figure(4);
subplot(2,1,1);
histogram(UAV_positions(:,3), 15, 'FaceColor', 'green', 'FaceAlpha', 0.7);
xlabel('UAV Height (m)'); ylabel('Number of UAVs');
title('Optimized UAV Height Distribution');
grid on;

%Calc per-cell user assignment data
users_per_cell_BS = zeros(N_cells, 1);
users_per_cell_UAV = zeros(N_cells, 1);

for cell_idx = 1:N_cells
    cell_users = find(user_cell_assignment == cell_idx);
    users_per_cell_BS(cell_idx) = sum(user_assignment(cell_users) == 0);
    users_per_cell_UAV(cell_idx) = sum(user_assignment(cell_users) == 1);
end

subplot(2,1,2);
bar_data = [users_per_cell_BS, users_per_cell_UAV];
bar(1:N_cells, bar_data, 'stacked');
xlabel('Cell Index'); ylabel('Number of Users');
title('User Assignment per Cell: BS vs UAV');
legend('BS Users', 'UAV Users', 'Location', 'best');
grid on;

%Fns for Enhanced User Distribution

function users = generate_clustered_users_in_hex(center, R, N_users, cluster_type)
    % Generate clustered users within hexagonal cell based on cluster type
    users = zeros(N_users, 2);
    
    switch cluster_type
        case 'hotspot'
            % Single high-density cluster near cell center
            cluster_centers = [center + [R*0.2*randn(), R*0.2*randn()]];
            cluster_weights = [1.0];
            cluster_spreads = [R*0.3];
            
        case 'commercial'
            % 2-3 medium-density clusters
            n_clusters = randi([2, 3]);
            cluster_centers = zeros(n_clusters, 2);
            for i = 1:n_clusters
                angle = 2*pi*rand();
                distance = R * 0.4 * rand();
                cluster_centers(i, :) = center + distance * [cos(angle), sin(angle)];
            end
            cluster_weights = ones(1, n_clusters) / n_clusters;
            cluster_spreads = R * 0.25 * ones(1, n_clusters);
            
        case 'residential'
            % 3-5 smaller clusters distributed across cell
            n_clusters = randi([3, 5]);
            cluster_centers = zeros(n_clusters, 2);
            for i = 1:n_clusters
                angle = 2*pi*rand();
                distance = R * (0.2 + 0.5*rand());
                cluster_centers(i, :) = center + distance * [cos(angle), sin(angle)];
            end
            cluster_weights = ones(1, n_clusters) / n_clusters;
            cluster_spreads = R * 0.2 * ones(1, n_clusters);
            
        case 'sparse'
            % Uniform distribution with slight clustering
            cluster_centers = [center];
            cluster_weights = [1.0];
            cluster_spreads = [R*0.8];
    end
    
    % Generate users based on cluster parameters
    for i = 1:N_users
        valid = false;
        attempts = 0;
        max_attempts = 100;
        
        while ~valid && attempts < max_attempts
            attempts = attempts + 1;
            
            % Select cluster based on weights
            cluster_idx = randsample(length(cluster_weights), 1, true, cluster_weights);
            cluster_center = cluster_centers(cluster_idx, :);
            cluster_spread = cluster_spreads(cluster_idx);
            
            % Generate user position around cluster center
            if strcmp(cluster_type, 'sparse')
                % More uniform distribution for sparse areas
                r = R * sqrt(rand());
                theta = 2 * pi * rand();
                x = center(1) + r * cos(theta);
                y = center(2) + r * sin(theta);
            else
                % Gaussian clustering around cluster center
                x = cluster_center(1) + cluster_spread * randn();
                y = cluster_center(2) + cluster_spread * randn();
            end
            
            % Check if point is inside hexagon
            if is_inside_hexagon_rotated([x, y], center, R, 120 * pi / 180)
                users(i, :) = [x, y];
                valid = true;
            end
        end
        
        % Fallback to uniform distribution if clustering fails
        if ~valid
            users(i, :) = generate_uniform_user_in_hex(center, R);
        end
    end
end

function user = generate_uniform_user_in_hex(center, R)
    % Fallback uniform user generation
    valid = false;
    while ~valid
        r = R * sqrt(rand());
        theta = 2 * pi * rand();
        x = center(1) + r * cos(theta);
        y = center(2) + r * sin(theta);
        
        if is_inside_hexagon_rotated([x, y], center, R, 120 * pi / 180)
            user = [x, y];
            valid = true;
        end
    end
end

function plot_clustered_users(all_users, user_cluster_types)
    % Plot users with different colors based on cluster type
    cluster_colors = containers.Map({'hotspot', 'commercial', 'residential', 'sparse'}, ...
                                   {'magenta', 'blue', 'green', 'cyan'});
    cluster_markers = containers.Map({'hotspot', 'commercial', 'residential', 'sparse'}, ...
                                    {'o', 's', '^', 'd'});
    
    hold on;
    for cluster_type = {'hotspot', 'commercial', 'residential', 'sparse'}
        cluster_type = cluster_type{1};
        cluster_users = all_users(strcmp(user_cluster_types, cluster_type), :);
        
        if ~isempty(cluster_users)
            plot(cluster_users(:,1), cluster_users(:,2), ...
                 cluster_markers(cluster_type), 'MarkerSize', 4, ...
                 'MarkerFaceColor', cluster_colors(cluster_type), ...
                 'MarkerEdgeColor', 'black', 'LineWidth', 0.5);
        end
    end
end


function UAV_positions = optimize_UAV_positions_enhanced(cell_centers, all_users, user_cell_assignment, R, user_cluster_types)
    % Enhanced UAV position optimization considering user clustering
    N_cells = size(cell_centers, 1);
    UAV_positions = zeros(N_cells, 3);
    
    for cell_idx = 1:N_cells
        fprintf('Optimizing UAV position for cell %d...\n', cell_idx);
        
        cell_users_idx = find(user_cell_assignment == cell_idx);
        if isempty(cell_users_idx)
            UAV_positions(cell_idx, :) = [cell_centers(cell_idx, :), 100];
            continue;
        end
        
        cell_users = all_users(cell_users_idx, :);
        cell_clusters = user_cluster_types(cell_users_idx);
        
        % Calculate weighted centroid based on cluster importance
        cluster_weights = containers.Map({'hotspot', 'commercial', 'residential', 'sparse'}, ...
                                        {3.0, 2.0, 1.2, 0.5});
        
        weighted_center = [0, 0];
        total_weight = 0;
        
        for i = 1:length(cell_users_idx)
            weight = cluster_weights(cell_clusters{i});
            weighted_center = weighted_center + weight * cell_users(i, :);
            total_weight = total_weight + weight;
        end
        
        if total_weight > 0
            weighted_center = weighted_center / total_weight;
        else
            weighted_center = cell_centers(cell_idx, :);
        end
        
        % Optimize around weighted centroid
        cell_center = cell_centers(cell_idx, :);
        x_bounds = [cell_center(1) - R*0.8, cell_center(1) + R*0.8];
        y_bounds = [cell_center(2) - R*0.8, cell_center(2) + R*0.8];
        h_bounds = [10, 120];
        
        options = optimoptions('ga', 'Display', 'off', 'MaxGenerations', 50);
        lb = [x_bounds(1), y_bounds(1), h_bounds(1)];
        ub = [x_bounds(2), y_bounds(2), h_bounds(2)];
        
        objective = @(pos) -evaluate_UAV_position_enhanced(pos, cell_users, cell_clusters, weighted_center);
        
        [optimal_pos, ~] = ga(objective, 3, [], [], [], [], lb, ub, [], options);
        UAV_positions(cell_idx, :) = optimal_pos;
    end
end

function score = evaluate_UAV_position_enhanced(UAV_pos, users, clusters, weighted_center)
    % Enhanced UAV position evaluation considering clustering
    if isempty(users)
        score = 0;
        return;
    end
    
    cluster_weights = containers.Map({'hotspot', 'commercial', 'residential', 'sparse'}, ...
                                    {3.0, 2.0, 1.2, 0.5});
    
    total_score = 0;
    P_tx_UAV = 23;
    fc_GHz = 2;
    
    % Distance penalty from weighted centroid
    centroid_distance = norm(UAV_pos(1:2) - weighted_center);
    centroid_penalty = centroid_distance / 1000; % Normalize
    
    for i = 1:size(users, 1)
        user_pos = users(i, :);
        cluster_type = clusters{i};
        cluster_weight = cluster_weights(cluster_type);
        
        d_2D = norm(user_pos - UAV_pos(1:2));
        d_3D = sqrt(d_2D^2 + UAV_pos(3)^2);
        
        [path_loss, ~] = calculate_UAV_path_loss(d_2D, UAV_pos(3), fc_GHz);
        signal_power = P_tx_UAV - path_loss;
        
        noise_power_dBm = -174 + 10*log10(10e6);
        SINR = signal_power - noise_power_dBm;
        
        if SINR > 0
            total_score = total_score + cluster_weight * SINR;
        end
    end
    
    score = (total_score / size(users, 1)) - centroid_penalty;
end

function analyze_performance_by_cluster(user_cluster_types, SINR_BS, SINR_UAV, throughput_BS, throughput_UAV, user_assignment)
    % Analyze network performance by cluster type
    fprintf('\n=== Performance Analysis by Cluster Type ===\n');
    
    cluster_types = unique(user_cluster_types);
    
    for i = 1:length(cluster_types)
        cluster_type = cluster_types{i};
        cluster_indices = strcmp(user_cluster_types, cluster_type);
        
        cluster_SINR_BS = SINR_BS(cluster_indices);
        cluster_SINR_UAV = SINR_UAV(cluster_indices);
        cluster_throughput_BS = throughput_BS(cluster_indices);
        cluster_throughput_UAV = throughput_UAV(cluster_indices);
        cluster_assignment = user_assignment(cluster_indices);
        
        bs_users = sum(cluster_assignment == 0);
        uav_users = sum(cluster_assignment == 1);
        total_users = length(cluster_assignment);
        
        fprintf('\n%s Cluster:\n', upper(cluster_type));
        fprintf('  Total users: %d\n', total_users);
        fprintf('  BS users: %d (%.1f%%)\n', bs_users, bs_users/total_users*100);
        fprintf('  UAV users: %d (%.1f%%)\n', uav_users, uav_users/total_users*100);
        fprintf('  Avg BS SINR: %.2f dB\n', mean(cluster_SINR_BS));
        fprintf('  Avg UAV SINR: %.2f dB\n', mean(cluster_SINR_UAV));
        fprintf('  Avg BS throughput: %.2f Mbps\n', mean(cluster_throughput_BS));
        fprintf('  Avg UAV throughput: %.2f Mbps\n', mean(cluster_throughput_UAV));
    end
end

function plot_cluster_performance_comparison(user_cluster_types, SINR_BS, SINR_UAV, throughput_BS, throughput_UAV, user_assignment)
    % Create comprehensive performance comparison plots
    cluster_types = {'hotspot', 'commercial', 'residential', 'sparse'};
    n_clusters = length(cluster_types);
    
    % Prepare data for plotting
    avg_sinr_bs = zeros(1, n_clusters);
    avg_sinr_uav = zeros(1, n_clusters);
    avg_throughput_bs = zeros(1, n_clusters);
    avg_throughput_uav = zeros(1, n_clusters);
    uav_usage_percent = zeros(1, n_clusters);
    
    for i = 1:n_clusters
        cluster_type = cluster_types{i};
        cluster_indices = strcmp(user_cluster_types, cluster_type);
        
        if sum(cluster_indices) > 0
            avg_sinr_bs(i) = mean(SINR_BS(cluster_indices));
            avg_sinr_uav(i) = mean(SINR_UAV(cluster_indices));
            avg_throughput_bs(i) = mean(throughput_BS(cluster_indices));
            avg_throughput_uav(i) = mean(throughput_UAV(cluster_indices));
            uav_usage_percent(i) = sum(user_assignment(cluster_indices) == 1) / sum(cluster_indices) * 100;
        end
    end
    
    subplot(2,2,1);
    bar(1:n_clusters, [avg_sinr_bs; avg_sinr_uav]', 'grouped');
    set(gca, 'XTickLabel', cluster_types);
    xlabel('Cluster Type'); ylabel('Average SINR (dB)');
    title('Average SINR by Cluster Type');
    legend('BS', 'UAV', 'Location', 'best');
    grid on;
    
    subplot(2,2,2);
    bar(1:n_clusters, [avg_throughput_bs; avg_throughput_uav]', 'grouped');
    set(gca, 'XTickLabel', cluster_types);
    xlabel('Cluster Type'); ylabel('Average Throughput (Mbps)');
    title('Average Throughput by Cluster Type');
    legend('BS', 'UAV', 'Location', 'best');
    grid on;
    
    subplot(2,2,3);
    bar(1:n_clusters, uav_usage_percent);
    set(gca, 'XTickLabel', cluster_types);
    xlabel('Cluster Type'); ylabel('UAV Usage (%)');
    title('UAV Usage Percentage by Cluster Type');
    grid on;
    
    subplot(2,2,4);
    user_counts = zeros(1, n_clusters);
    for i = 1:n_clusters
        user_counts(i) = sum(strcmp(user_cluster_types, cluster_types{i}));
    end
    pie(user_counts, cluster_types);
    title('User Distribution by Cluster Type');
end


function SINR_dB = calculate_SINR_BS(user_pos, serving_BS, all_BS, user_cell_assignment, P_tx, path_loss_exp)
    d_serving = norm(user_pos - serving_BS);
    path_loss_serving = calculate_path_loss(d_serving, path_loss_exp);
    signal_power = P_tx - path_loss_serving;
    
    interference_power = 0;
    noise_power_dBm = -174 + 10*log10(10e6);
    
    signal_power_linear = 10^((signal_power - 30)/10);
    noise_power_linear = 10^((noise_power_dBm - 30)/10);
    
    SINR_linear = signal_power_linear / (interference_power + noise_power_linear);
    SINR_dB = 10 * log10(SINR_linear);
end

function SINR_dB = calculate_SINR_UAV(user_pos, serving_UAV, all_UAV, user_cell_assignment, P_tx, fc_GHz)
    d_2D = norm(user_pos - serving_UAV(1:2));
    %d_3D = sqrt(d_2D^2 + serving_UAV(3)^2);
    
    [path_loss, ~] = calculate_UAV_path_loss(d_2D, serving_UAV(3), fc_GHz);
    signal_power = P_tx - path_loss;
    
    noise_power_dBm = -174 + 10*log10(10e6);
    
    signal_power_linear = 10^((signal_power - 30)/10);
    noise_power_linear = 10^((noise_power_dBm - 30)/10);
    
    SINR_linear = signal_power_linear / noise_power_linear;
    SINR_dB = 10 * log10(SINR_linear);
end

function [path_loss_dB, is_LoS] = calculate_UAV_path_loss(d_2D, height, fc_GHz)
    d_3D = sqrt(d_2D^2 + height^2);
    
    if height > 100
        P_LoS = 1;
    elseif d_2D <= max(460*log10(height) - 700, 18)
        P_LoS = 1;
    else
        d1 = max(460*log10(height) - 700, 18);
        p1 = 4300*log10(height) - 3800;
        P_LoS = d1/d_2D + exp(-d_2D/p1) * (1 - d1/d_2D);
    end
    
    is_LoS = rand() < P_LoS;
    
    if is_LoS
        path_loss_dB = 28.0 + 22*log10(d_3D) + 20*log10(fc_GHz);
    else
        path_loss_dB = -17.5 + (46 - 7*log10(height))*log10(d_3D) + 20*log10(40*pi*fc_GHz/3);
    end
end

function path_loss_dB = calculate_path_loss(distance, path_loss_exp)
    if distance == 0
        path_loss_dB = 0;
    else
        f_GHz = 2;
        PL_1m = 32.44 + 20 * log10(f_GHz);
        path_loss_dB = PL_1m + 10 * path_loss_exp * log10(distance);
    end
end

function centers = generate_hex_centers_rotated(R)
    ISD = sqrt(3) * R;
    rotation_angle = 120 * pi / 180;
    rotation_matrix = [cos(rotation_angle) -sin(rotation_angle); 
                       sin(rotation_angle)  cos(rotation_angle)];
    
    centers = zeros(19, 2);
    centers(1, :) = [0, 0];
    
    thetaRing1 = 0:pi/3:(5*pi/3);
    coordinatesRing1 = [cos(thetaRing1); sin(thetaRing1)]';
    centers(2:7, :) = ISD * coordinatesRing1;
    
    thetaRing2 = pi/6 + (0:pi/3:(5*pi/3));
    coordinatesRing2 = [cos(thetaRing2); sin(thetaRing2)]';
    centers(8:13, :) = ISD * sqrt(3) * coordinatesRing2;
    centers(14:19, :) = 2 * ISD * coordinatesRing1;
    
    for i = 1:19
        rotated_center = rotation_matrix * centers(i, :)';
        centers(i, :) = rotated_center';
    end
end

function plot_hexagonal_cells_rotated(centers, R)
    rotation_angle = 90 * pi / 180;
    hold on;
    for i = 1:size(centers, 1)
        hex_x = []; hex_y = [];
        for angle = 0:60:300
            angle_rad = deg2rad(angle) + rotation_angle;
            hex_x = [hex_x, centers(i,1) + R * cos(angle_rad)];
            hex_y = [hex_y, centers(i,2) + R * sin(angle_rad)];
        end
        hex_x = [hex_x, hex_x(1)]; hex_y = [hex_y, hex_y(1)];
        plot(hex_x, hex_y, 'w-', 'LineWidth', 1.5);
    end
end

function inside = is_inside_hexagon_rotated(point, center, R, rotation_angle)
    p = point - center;
    inside = true;
    apothem = R * cos(pi/6);
    
    for angle = 0:60:300
        angle_rad = deg2rad(angle + 30) + rotation_angle;
        normal = [cos(angle_rad), sin(angle_rad)];
        
        if dot(p, normal) > apothem
            inside = false;
            break;
        end
    end
end
