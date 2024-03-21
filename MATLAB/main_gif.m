% [Project]        DMP Comparison - Video Generation
% [Author]         Moses C. Nah
% [Creation Date]  Monday, Oct. 23th, 2022
%
% [Emails]         Moses C. Nah   : mosesnah@mit.edu

%% (--) INITIALIZATION

clear; close all; clc; workspace;

% Add the Libraries of 
cd( fileparts( matlab.desktop.editor.getActiveFilename ) );     
addpath( 'MATLAB_Library/myUtils', 'MATLAB_Library/myGraphics' )
myFigureConfig( 'fontsize',  20, ...
               'LineWidth',  10, ...
           'AxesLineWidth', 1.5, ...     For Grid line, axes line width etc.
              'markerSize',  25    )  
             
% Setting color structure 'c' as global variable
global c                                                                   
c  = myColor(); 

%% (--) Example 1: Goal-directed Discrete Movement in Joint-space
% Figure 3 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches
clear data*; clc; close all;

fs = 40;
lw1 = 6; lw2= 9;

% Folder name
dir_name = '../main/example1_joint_discrete/data/';

file_name_DMP = [ dir_name, 'DMP.mat' ];
file_name_EDA = [ dir_name, 'EDA_Kq150_Bq50.mat' ];

data_DMP = load( file_name_DMP );
data_EDA = load( file_name_EDA );

% The robot image
q_abs = cumsum( data_EDA.q_arr , 2 );
x_EDA = cumsum( cos( q_abs ), 2 );
y_EDA = cumsum( sin( q_abs ), 2 );

% The robot image
q0_abs = cumsum( data_EDA.q0_arr , 2 );
x0_EDA = cumsum( cos( q0_abs ), 2 );
y0_EDA = cumsum( sin( q0_abs ), 2 );


% The robot image
q_abs = cumsum( data_DMP.q_arr , 2 );
x_DMP = cumsum( cos( q_abs ), 2 );
y_DMP = cumsum( sin( q_abs ), 2 );

f = figure( ); 

a1 = subplot( 1, 2, 1 );
hold on
gDMP = plot( a1, [ 0, x_DMP( 1, : ) ], [ 0, y_DMP( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gDMP_SH = scatter( a1, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gDMP_EL = scatter( a1, x_DMP( 1, 1 ), y_DMP( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gDMP_EE = scatter( a1, x_DMP( 1, 2 ), y_DMP( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );

set( a1, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
          'xticklabel', { }, 'yticklabel', { } ) 
title( a1, 'Dynamic Movement Primitives (DMP)')

a2 = subplot( 1, 2, 2 );
hold on
gEDA0 = plot( a2, [ 0, x0_EDA( 1, : ) ], [ 0, y0_EDA( 1, : ) ], 'color', 0.3*ones(1,3), 'linewidth', 4 );
gEDA0_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA0_EL = scatter( a2, x0_EDA( 1, 1 ), y0_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5  );
gEDA0_EE = scatter( a2, x0_EDA( 1, 2 ), y0_EDA( 1, 2 ), 2400, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

gEDA = plot( a2, [ 0, x_EDA( 1, : ) ], [ 0, y_EDA( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA_EL = scatter( a2, x_EDA( 1, 1 ), y_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA_EE = scatter( a2, x_EDA( 1, 2 ), y_EDA( 1, 2 ), 2400, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6 );


set( a2, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
    'xticklabel', { }, 'yticklabel', { } ) 
title( a2, 'Elementary Dynamic Actions (EDA)')

% Time per frame
fps = 60;
t_arr = data_DMP.t_arr;
T = max( data_DMP.t_arr );
numFrames    = round( T*fps*1.5);
timePerFrame = T / numFrames;

delayTime = 1./fps;

T_min = 0.0;
T_max = 3.0;

n_frame = 1;
for frameIdx = 1:numFrames

    % Current time for this frame
    currentTime = (frameIdx - 1) * timePerFrame;

    
    if currentTime <= T_min
        continue
    end
    % Find the closest time in your timeArray (or interpolate if needed)
    [ ~,  iidx ] = min( abs( t_arr - currentTime ) );


    set( gDMP, 'xData', [ 0, x_DMP( iidx, : )],  'yData', [ 0, y_DMP( iidx, : )] )
    set( gDMP_EL, 'XData', x_DMP( iidx, 1 ), 'YData', y_DMP( iidx, 1 ) );
    set( gDMP_EE, 'XData', x_DMP( iidx, 2 ), 'YData', y_DMP( iidx, 2 ) );    
    
    set( gEDA, 'xData', [ 0, x_EDA( iidx, : )],  'yData', [ 0, y_EDA( iidx, : )] )
    set( gEDA_EL, 'XData', x_EDA( iidx, 1 ), 'YData', y_EDA( iidx, 1 ) );
    set( gEDA_EE, 'XData', x_EDA( iidx, 2 ), 'YData', y_EDA( iidx, 2 ) );    

    set( gEDA0, 'xData', [ 0, x0_EDA( iidx, : )],  'yData', [ 0, y0_EDA( iidx, : )] )
    set( gEDA0_EL, 'XData', x0_EDA( iidx, 1 ), 'YData', y0_EDA( iidx, 1 ) );
    set( gEDA0_EE, 'XData', x0_EDA( iidx, 2 ), 'YData', y0_EDA( iidx, 2 ) );    


    % Capture the plot as an image 
    frame = getframe( f ); 
    im    = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example1.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example1.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end

