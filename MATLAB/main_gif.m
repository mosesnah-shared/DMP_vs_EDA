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

%% (--) Example 1a: Goal-directed Discrete Movement in Joint-space
% Figure 3 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches
clear data*; clc; close all;

fs = 40;
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
set( f,  'Units', 'pixels' );

a1 = subplot( 1, 2, 1 );
hold on
gDMP = plot( a1, [ 0, x_DMP( 1, : ) ], [ 0, y_DMP( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gDMP_SH = scatter( a1, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gDMP_EL = scatter( a1, x_DMP( 1, 1 ), y_DMP( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gDMP_EE = scatter( a1, x_DMP( 1, 2 ), y_DMP( 1, 2 ), 2400, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6 );

set( a1, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
          'xticklabel', { }, 'yticklabel', { } ) 
title( a1, 'Dynamic Movement Primitives (DMP)')

a2 = subplot( 1, 2, 2 );
hold on
gEDA0 = plot( a2, [ 0, x0_EDA( 1, : ) ], [ 0, y0_EDA( 1, : ) ], 'color', 0.3*ones(1,3), 'linewidth', 4 );
gEDA0_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA0_EL = scatter( a2, x0_EDA( 1, 1 ), y0_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5  );
gEDA0_EE = scatter( a2, x0_EDA( 1, 2 ), y0_EDA( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

gEDA = plot( a2, [ 0, x_EDA( 1, : ) ], [ 0, y_EDA( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA_EL = scatter( a2, x_EDA( 1, 1 ), y_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA_EE = scatter( a2, x_EDA( 1, 2 ), y_EDA( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );


set( a2, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
    'xticklabel', { }, 'yticklabel', { } ) 
title( a2, 'Elementary Dynamic Actions (EDA)')

% Time per frame
fps = 30;

im_ratio = 0.25;
t_arr = data_DMP.t_arr;
T = max( data_DMP.t_arr );
numFrames    = round( T*fps*1.5);
timePerFrame = T / numFrames;

delayTime = 1./fps;

T_min = 0.0;
T_max = 2.0;

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
    im = imresize(im, im_ratio );

    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example1a.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example1a.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end

%% (--) Example 1b: Goal-directed Discrete Movement in Joint-space, EDA, Different Parameters
clear data*; clc; close all;

fs = 40;

% Folder name
dir_name = '../main/example1_joint_discrete/data/';

file_name_EDA1 = [ dir_name,  'EDA_Kq150_Bq50.mat' ];
file_name_EDA2 = [ dir_name, 'EDA_Kq1200_Bq50.mat' ];

data_EDA1 = load( file_name_EDA1 );
data_EDA2 = load( file_name_EDA2 );

% The robot image
q_abs = cumsum( data_EDA1.q_arr , 2 );
x_EDA1 = cumsum( cos( q_abs ), 2 );
y_EDA1 = cumsum( sin( q_abs ), 2 );

% The robot image
q0_abs = cumsum( data_EDA1.q0_arr , 2 );
x0_EDA1 = cumsum( cos( q0_abs ), 2 );
y0_EDA1 = cumsum( sin( q0_abs ), 2 );


% The robot image
q_abs = cumsum( data_EDA2.q_arr , 2 );
x_EDA2 = cumsum( cos( q_abs ), 2 );
y_EDA2 = cumsum( sin( q_abs ), 2 );

% The robot image
q0_abs = cumsum( data_EDA2.q0_arr , 2 );
x0_EDA2 = cumsum( cos( q0_abs ), 2 );
y0_EDA2 = cumsum( sin( q0_abs ), 2 );

f = figure( ); 

a1 = subplot( 1, 2, 1 );
hold on

gEDA1_0 = plot( a1, [ 0, x0_EDA1( 1, : ) ], [ 0, y0_EDA1( 1, : ) ], 'color', 0.3*ones(1,3), 'linewidth', 4 );
gEDA1_SH0 = scatter( a1, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA1_EL0 = scatter( a1, x0_EDA1( 1, 1 ), y0_EDA1( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA1_EE0 = scatter( a1, x0_EDA1( 1, 2 ), y0_EDA1( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );


gEDA1 = plot( a1, [ 0, x_EDA1( 1, : ) ], [ 0, y_EDA1( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA1_SH = scatter( a1, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA1_EL = scatter( a1, x_EDA1( 1, 1 ), y_EDA1( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA1_EE = scatter( a1, x_EDA1( 1, 2 ), y_EDA1( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );

set( a1, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
          'xticklabel', { }, 'yticklabel', { } ) 
title( a1, 'EDA with Low $\mathbf{K}_q$')

a2 = subplot( 1, 2, 2 );
hold on

gEDA2_0 = plot( a2, [ 0, x0_EDA2( 1, : ) ], [ 0, y0_EDA2( 1, : ) ], 'color', 0.3*ones(1,3), 'linewidth', 4 );
gEDA2_SH0 = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA2_EL0 = scatter( a2, x0_EDA2( 1, 1 ), y0_EDA2( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 , 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );
gEDA2_EE0 = scatter( a2, x0_EDA2( 1, 2 ), y0_EDA2( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 , 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

gEDA2 = plot( a2, [ 0, x_EDA2( 1, : ) ], [ 0, y_EDA2( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA2_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA2_EL = scatter( a2, x_EDA2( 1, 1 ), y_EDA2( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA2_EE = scatter( a2, x_EDA2( 1, 2 ), y_EDA2( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );



set( a2, 'xlim', [-0.3, 2.3] , 'ylim', [-0.2, 2.4], 'xtick', 0:1:2, 'ytick', 0:1:2, 'fontsize', 1.2*fs, ...
    'xticklabel', { }, 'yticklabel', { } ) 
title( a2, 'EDA with High $\mathbf{K}_q$')

% Time per frame
fps = 60;
t_arr = data_EDA1.t_arr;
T = max( data_EDA1.t_arr );
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


    set( gEDA1, 'xData', [ 0, x_EDA1( iidx, : )],  'yData', [ 0, y_EDA1( iidx, : )] )
    set( gEDA1_EL, 'XData', x_EDA1( iidx, 1 ), 'YData', y_EDA1( iidx, 1 ) );
    set( gEDA1_EE, 'XData', x_EDA1( iidx, 2 ), 'YData', y_EDA1( iidx, 2 ) );    

    set( gEDA1_0, 'xData', [ 0, x0_EDA1( iidx, : )],  'yData', [ 0, y0_EDA1( iidx, : )] )
    set( gEDA1_EL0, 'XData', x0_EDA1( iidx, 1 ), 'YData', y0_EDA1( iidx, 1 ) );
    set( gEDA1_EE0, 'XData', x0_EDA1( iidx, 2 ), 'YData', y0_EDA1( iidx, 2 ) );    


    set( gEDA2, 'xData', [ 0, x_EDA2( iidx, : )],  'yData', [ 0, y_EDA2( iidx, : )] )
    set( gEDA2_EL, 'XData', x_EDA2( iidx, 1 ), 'YData', y_EDA2( iidx, 1 ) );
    set( gEDA2_EE, 'XData', x_EDA2( iidx, 2 ), 'YData', y_EDA2( iidx, 2 ) );    

    set( gEDA2_0, 'xData', [ 0, x0_EDA2( iidx, : )],  'yData', [ 0, y0_EDA2( iidx, : )] )
    set( gEDA2_EL0, 'XData', x0_EDA2( iidx, 1 ), 'YData', y0_EDA2( iidx, 1 ) );
    set( gEDA2_EE0, 'XData', x0_EDA2( iidx, 2 ), 'YData', y0_EDA2( iidx, 2 ) );    


    % Capture the plot as an image 
    frame = getframe( f ); 
    im    = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example1b.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example1b.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end

%% (--) Example 2a: Goal-directed Discrete Movement in Task-space

% Figure 3 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches
clear data*; clc; close all;

fs = 40;
% Folder name
dir_name = '../main/example2_task_discrete/data/';

file_name_DMP = [ dir_name, 'DMP.mat' ];
file_name_EDA = [ dir_name, 'EDA_Kp60_Bp20.mat' ];

data_DMP = load( file_name_DMP );
data_EDA = load( file_name_EDA );

g_start = data_EDA.p0_arr(    1, : );
g_end   = data_EDA.p0_arr( end, : );


% The robot image
q_abs = cumsum( data_EDA.q_arr , 2 );
x_EDA = cumsum( cos( q_abs ), 2 );
y_EDA = cumsum( sin( q_abs ), 2 );

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
gDMP_EE = scatter( a1, x_DMP( 1, 2 ), y_DMP( 1, 2 ), 2400, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6 );

gDMP0_EE = scatter( a1, data_DMP.p_des( 1, 2 ), data_DMP.p_des( 2, 2 ), 1200, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
title( a1, 'Dynamic Movement Primitives (DMP)')

% Start and End Location
scatter( a1, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a1, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )

text( a1, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a1, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )
plot( a1, data_EDA.p0_arr( :, 1 ), data_EDA.p0_arr( :, 2 ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

a2 = subplot( 1, 2, 2 );
hold on

gEDA = plot( a2, [ 0, x_EDA( 1, : ) ], [ 0, y_EDA( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA_EL = scatter( a2, x_EDA( 1, 1 ), y_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA_EE = scatter( a2, x_EDA( 1, 2 ), y_EDA( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );

gEDA0_EE = scatter( a2, data_EDA.p0_arr( 1, 1 ), data_EDA.p0_arr( 1, 2 ), 1200, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );


set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
title( a2, 'Elementary Dynamic Actions (EDA)')
% Start and End Location
scatter( a2, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a2, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
plot( a2, data_EDA.p0_arr( :, 1 ), data_EDA.p0_arr( :, 2 ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

text( a2, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a2, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )

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
    set( gDMP0_EE, 'XdAta', data_DMP.p_des( 1, iidx+1 ), 'YData', data_DMP.p_des( 2, iidx+1 ))

    set( gEDA, 'xData', [ 0, x_EDA( iidx, : )],  'yData', [ 0, y_EDA( iidx, : )] )
    set( gEDA_EL, 'XData', x_EDA( iidx, 1 ), 'YData', y_EDA( iidx, 1 ) );
    set( gEDA_EE, 'XData', x_EDA( iidx, 2 ), 'YData', y_EDA( iidx, 2 ) );    
    set( gEDA0_EE, 'XdAta',  data_EDA.p0_arr( iidx, 1),  'YData', data_EDA.p0_arr( iidx, 2 ) )


    % Capture the plot as an image 
    frame = getframe( f ); 
    im    = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example2a.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example2a.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end


%% (--) Example 2b: Goal-directed Discrete Movement in Task-space, with Singularity

% Figure 3 on the manuscript, Nah, Lachner and Hogan 2024
% Robot Control based on Motor Primitives — A Comparison of Two Approaches
clear data*; clc; close all;

fs = 40;
% Folder name
dir_name = '../main/example2_task_discrete/data/';

file_name_DMP = [ dir_name, 'DMP_sing_damped.mat' ];
file_name_EDA = [ dir_name, 'EDA_Kp60_Bp20_sing.mat' ];

data_DMP = load( file_name_DMP );
data_EDA = load( file_name_EDA );

g_start = data_EDA.p0_arr(    1, : );
g_end   = data_EDA.p0_arr( end, : );


% The robot image
q_abs = cumsum( data_EDA.q_arr , 2 );
x_EDA = cumsum( cos( q_abs ), 2 );
y_EDA = cumsum( sin( q_abs ), 2 );

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
gDMP_EE = scatter( a1, x_DMP( 1, 2 ), y_DMP( 1, 2 ), 2400, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6 );

gDMP0_EE = scatter( a1, data_DMP.p_des( 1, 2 ), data_DMP.p_des( 2, 2 ), 1200, 'markerfacecolor', c.blue, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
title( a1, 'Dynamic Movement Primitives (DMP)')

% Start and End Location
scatter( a1, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a1, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )

text( a1, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a1, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )
plot( a1, data_DMP.p_des( 1, : ), data_DMP.p_des( 2, : ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

a2 = subplot( 1, 2, 2 );
hold on

gEDA = plot( a2, [ 0, x_EDA( 1, : ) ], [ 0, y_EDA( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA_EL = scatter( a2, x_EDA( 1, 1 ), y_EDA( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA_EE = scatter( a2, x_EDA( 1, 2 ), y_EDA( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );

gEDA0_EE = scatter( a2, data_EDA.p0_arr( 1, 1 ), data_EDA.p0_arr( 1, 2 ), 1200, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );


set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
title( a2, 'Elementary Dynamic Actions (EDA)')
% Start and End Location
scatter( a2, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a2, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
plot( a2, data_DMP.p_des( 1, : ), data_DMP.p_des( 2, : ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

text( a2, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a2, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )

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
    set( gDMP0_EE, 'XdAta', data_DMP.p_des( 1, iidx+1 ), 'YData', data_DMP.p_des( 2, iidx+1 ))

    set( gEDA, 'xData', [ 0, x_EDA( iidx, : )],  'yData', [ 0, y_EDA( iidx, : )] )
    set( gEDA_EL, 'XData', x_EDA( iidx, 1 ), 'YData', y_EDA( iidx, 1 ) );
    set( gEDA_EE, 'XData', x_EDA( iidx, 2 ), 'YData', y_EDA( iidx, 2 ) );    
    set( gEDA0_EE, 'XdAta',  data_EDA.p0_arr( iidx, 1),  'YData', data_EDA.p0_arr( iidx, 2 ) )


    % Capture the plot as an image 
    frame = getframe( f ); 
    im    = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example2b.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example2b.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end

%% (--) Example 2c: Goal-directed Discrete Movement in Task-space, EDA, Different Parameters


clear data*; clc; close all;

fs = 40;

% Folder name
dir_name = '../main/example2_task_discrete/data/';

file_name_EDA1 = [ dir_name,  'EDA_Kp60_Bp20.mat' ];
file_name_EDA2 = [ dir_name, 'EDA_Kp300_Bp20.mat' ];

data_EDA1 = load( file_name_EDA1 );
data_EDA2 = load( file_name_EDA2 );

% The robot image
q_abs = cumsum( data_EDA1.q_arr , 2 );
x_EDA1 = cumsum( cos( q_abs ), 2 );
y_EDA1 = cumsum( sin( q_abs ), 2 );


% The robot image
q_abs = cumsum( data_EDA2.q_arr , 2 );
x_EDA2 = cumsum( cos( q_abs ), 2 );
y_EDA2 = cumsum( sin( q_abs ), 2 );

f = figure( ); 

a1 = subplot( 1, 2, 1 );
hold on

gEDA1 = plot( a1, [ 0, x_EDA1( 1, : ) ], [ 0, y_EDA1( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA1_SH = scatter( a1, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA1_EL = scatter( a1, x_EDA1( 1, 1 ), y_EDA1( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA1_EE = scatter( a1, x_EDA1( 1, 2 ), y_EDA1( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );

gEDA1_0_EE = scatter( a1, data_EDA1.p0_arr( 1, 1 ), data_EDA1.p0_arr( 1, 2 ), 1200, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );


set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
% title( a1, 'Elementary Dynamic Actions (EDA)')
% Start and End Location
scatter( a1, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a1, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
plot( a1, data_EDA1.p0_arr( :, 1 ), data_EDA1.p0_arr( :, 2 ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )
text( a1, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a1, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )


title( a1, 'EDA with Low $\mathbf{K}_p$')

a2 = subplot( 1, 2, 2 );
hold on

gEDA2 = plot( a2, [ 0, x_EDA2( 1, : ) ], [ 0, y_EDA2( 1, : ) ], 'color', c.black, 'linewidth', 8 );
gEDA2_SH = scatter( a2, 0, 0, 2400, 'markerfacecolor', c.black, 'markeredgecolor', c.black, 'markerfacecolor', c.white, 'linewidth', 6 );
gEDA2_EL = scatter( a2, x_EDA2( 1, 1 ), y_EDA2( 1, 1 ), 2400, 'markeredgecolor', c.black, 'markerfacecolor', c.white,'linewidth', 6 );
gEDA2_EE = scatter( a2, x_EDA2( 1, 2 ), y_EDA2( 1, 2 ), 2400, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6 );
gEDA2_0_EE = scatter( a2, data_EDA2.p0_arr( 1, 1 ), data_EDA2.p0_arr( 1, 2 ), 1200, 'markerfacecolor', c.orange, 'markeredgecolor', c.black, 'linewidth', 6, 'markerfacealpha', 0.5, 'markeredgealpha', 0.5 );

set( gca, 'xlim', [-1.1, 1.1] , 'ylim', [-0.2, 2.4], 'xtick', [-1.0, 0.0, 1.0], 'ytick', [0.0, 1.0, 2.0], 'fontsize', 1.2*fs, 'xticklabel', { }, 'yticklabel', { } ) 
% title( a1, 'Elementary Dynamic Actions (EDA)')
% Start and End Location
scatter( a2, 0, g_start( 2 ), 300, 'o', 'markerfacecolor', c.pink_sunset, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
scatter( a2, 0,   g_end( 2 ), 300, 'square', 'markerfacecolor', c.white, 'markeredgecolor', c.black, 'markerfacealpha', 1.0, 'linewidth', 6 )
plot( a2, data_EDA1.p0_arr( :, 1 ), data_EDA1.p0_arr( :, 2 ), 'linewidth', 4, 'color', c.black, 'linestyle',  '--' )

text( a2, -0.5, g_start( 2 ), 'Start $\mathbf{p}_i$' , 'fontsize', fs)
text( a2, -0.5, g_end( 2 ), 'Goal $\mathbf{g}$'   , 'fontsize', fs )


title( a2, 'EDA with High $\mathbf{K}_p$')

% Time per frame
fps = 15;
im_ratio = 0.5;
t_arr = data_EDA1.t_arr;
T = max( data_EDA1.t_arr );
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


    set( gEDA1, 'xData', [ 0, x_EDA1( iidx, : )],  'yData', [ 0, y_EDA1( iidx, : )] )
    set( gEDA1_EL, 'XData', x_EDA1( iidx, 1 ), 'YData', y_EDA1( iidx, 1 ) );
    set( gEDA1_EE, 'XData', x_EDA1( iidx, 2 ), 'YData', y_EDA1( iidx, 2 ) );    
    set( gEDA1_0_EE,  'XData', data_EDA1.p0_arr( iidx, 1 ), 'YData', data_EDA1.p0_arr( iidx, 2 ) )


    set( gEDA2, 'xData', [ 0, x_EDA2( iidx, : )],  'yData', [ 0, y_EDA2( iidx, : )] )
    set( gEDA2_EL, 'XData', x_EDA2( iidx, 1 ), 'YData', y_EDA2( iidx, 1 ) );
    set( gEDA2_EE, 'XData', x_EDA2( iidx, 2 ), 'YData', y_EDA2( iidx, 2 ) );    
    set( gEDA2_0_EE,  'XData', data_EDA2.p0_arr( iidx, 1 ), 'YData', data_EDA2.p0_arr( iidx, 2 ) )


    % Capture the plot as an image 
    frame = getframe( f ); 
    im    = frame2im(frame); 
    im = imresize( im, im_ratio );
    [imind,cm] = rgb2ind(im,256); 
    
    % Write to the GIF File 
    if n_frame == 1
        imwrite( imind, cm,'./gifs/example2c.gif', 'gif', 'Loopcount',inf, 'DelayTime', delayTime);
    else 
        imwrite( imind, cm,'./gifs/example2c.gif', 'gif', 'WriteMode','append','DelayTime', delayTime);
    end 

    n_frame = n_frame + 1;
end