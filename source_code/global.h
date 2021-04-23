#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32
#define coul2Klevin 1.6710095663e+05



// upgrade solution 3 for GPU Vext on the initial plane
__global__
void Vext_cal(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device, 
                double *charge_adsorbate_device, 
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *charge_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device, double *damping_a_device, 

                int *direction_device, 


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *cal_a_device, double *cal_b_device, double *cal_c_device,
                double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device,
                double *loc_x_device, double *loc_y_device, double *loc_z_device,
                double *vector_adsorbate_x_rot_device, double *vector_adsorbate_y_rot_device, double *vector_adsorbate_z_rot_device,
                double *adsorbate_cart_x_rot_device, double *adsorbate_cart_y_rot_device, double *adsorbate_cart_z_rot_device, 
                double *modify_frame_a_device, double *modify_frame_b_device, double *modify_frame_c_device,
                double *minimum_distance_device,
                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, jj;
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                index_a_device[i] = 0;
                index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 2:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = 0;
                index_c_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 3:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = 0;
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
        }

        cal_a_device[i] = index_a_device[i]*delta_grid_device[0];
        cal_b_device[i] = index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = index_gamma_device[i]*delta_angle_device[0]/180*PI;

        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


        vector_adsorbate_x_rot_device[i] = rotate_moleucle_x_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_y_rot_device[i] = rotate_moleucle_y_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_z_rot_device[i] = rotate_moleucle_z_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);

        adsorbate_cart_x_rot_device[i] = loc_x_device[i]+vector_adsorbate_x_rot_device[i];
        adsorbate_cart_y_rot_device[i] = loc_y_device[i]+vector_adsorbate_y_rot_device[i];
        adsorbate_cart_z_rot_device[i] = loc_z_device[i]+vector_adsorbate_z_rot_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] + times_z_device[0];
        }
        else if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] - times_z_device[0];
        }
        else
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]];
        }



        // y-direction
        if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] + times_y_device[0];
        }
        else if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] - times_y_device[0];
        }
        else
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]];
        }

        // x-direction
        if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] + times_x_device[0];
        }
        else if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] - times_x_device[0];
        }
        else
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]];
        }

        minimum_distance_device[i] = cal_dis_device(adsorbate_cart_x_rot_device[i], adsorbate_cart_y_rot_device[i], adsorbate_cart_z_rot_device[i], 
                    frac2car_x_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_a_device), 
                    frac2car_y_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_b_device), 
                    frac2car_z_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_c_device));

        if (minimum_distance_device[i] < 
            (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1))
        {
            minimum_distance_device[i] = (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1);
        }

        if (minimum_distance_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0])

            + cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_device[i]], charge_frame_device[index_frame_device[i]], 
                damping_a_device[0], minimum_distance_device[i]) 
            - cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_device[i]], charge_frame_device[index_frame_device[i]], 
                damping_a_device[0], cutoff_device[0]);
        }
        
    }

}








//
__global__
void ini_string_1(int *N_string_device, double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 
    int *temp_add_frame_device, int *N_atom_adsorbate_device,


    int *direction_device, 
    cub::KeyValuePair<int, double> *min_value_index_device, double *s0_a_device, double *s0_b_device, double *s0_c_device,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int i, j, jj;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<N_string_device[0]; i+=stride)
    {
        switch (direction_device[0])
        {
            // initialize string along the x-axis direction
            case 1:
                s0_a_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_b_device[i] = cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_c_device[i] = cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 2:
                s0_a_device[i] = cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_b_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_c_device[i] = cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 3:
                s0_a_device[i] = cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_b_device[i] = cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_c_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
        }
        
        
    }

}





// copy the initila and the last point
__global__
void copy_ini_upgrade(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    int *direction_device,

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(2*6); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
            case 2:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;

            case 3:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
        }
    }

}









// copy the middle points on the initial string 
__global__
void copy_ini_middle_upgrade(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    int *direction_device,

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    int *i_cal_device,

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(1*6); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
            case 2:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];;
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;

            case 3:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
        }
        
        
        
    }

}









// solution for GPU Vext on the initial string
__global__
void Vext_cal_ini(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
                double *charge_adsorbate_device, 
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *charge_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device, double *damping_a_device, 


                int *direction_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *limit_transition_frac_device, double *limit_rotation_angle_device,
                double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
                double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device,

                int *i_cal_device, int *num_inidividual_ini_extra_device,



                double *cal_a_device, double *cal_b_device, double *cal_c_device,
                double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device,
                double *loc_x_device, double *loc_y_device, double *loc_z_device,
                double *vector_adsorbate_x_rot_device, double *vector_adsorbate_y_rot_device, double *vector_adsorbate_z_rot_device,
                double *adsorbate_cart_x_rot_device, double *adsorbate_cart_y_rot_device, double *adsorbate_cart_z_rot_device, 
                double *modify_frame_a_device, double *modify_frame_b_device, double *modify_frame_c_device,
                double *minimum_distance_device,
                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, jj;
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                index_a_device[i] = 0;
                index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 2:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = 0;
                index_c_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 3:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = 0;
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
        }

        switch (direction_device[0])
        {
            case 1:
                cal_a_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                cal_b_device[i] = ini_minimum_string_b_device[0] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
                cal_c_device[i] = ini_minimum_string_c_device[0] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];
                break;
            case 2:
                cal_a_device[i] = ini_minimum_string_a_device[0] - limit_transition_frac_device[0] + index_a_device[i]*delta_grid_device[0];
                cal_b_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                cal_c_device[i] = ini_minimum_string_c_device[0] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];
                break;
            case 3:
                cal_a_device[i] = ini_minimum_string_a_device[0] - limit_transition_frac_device[0] + index_a_device[i]*delta_grid_device[0];
                cal_b_device[i] = ini_minimum_string_b_device[0] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
                cal_c_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                break;
        }

        rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = ini_minimum_string_beta_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;

        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);

        vector_adsorbate_x_rot_device[i] = rotate_moleucle_x_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_y_rot_device[i] = rotate_moleucle_y_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_z_rot_device[i] = rotate_moleucle_z_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);

        adsorbate_cart_x_rot_device[i] = loc_x_device[i]+vector_adsorbate_x_rot_device[i];
        adsorbate_cart_y_rot_device[i] = loc_y_device[i]+vector_adsorbate_y_rot_device[i];
        adsorbate_cart_z_rot_device[i] = loc_z_device[i]+vector_adsorbate_z_rot_device[i];

        V_result_device[i] = 0;

        // z-direction
        if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] + times_z_device[0];
        }
        else if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] - times_z_device[0];
        }
        else
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]];
        }

        // y-direction
        if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] + times_y_device[0];
        }
        else if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] - times_y_device[0];
        }
        else
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]];
        }

        // x-direction
        if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] + times_x_device[0];
        }
        else if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] - times_x_device[0];
        }
        else
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]];
        }

        minimum_distance_device[i] = cal_dis_device(adsorbate_cart_x_rot_device[i], adsorbate_cart_y_rot_device[i], adsorbate_cart_z_rot_device[i], 
                    frac2car_x_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_a_device), 
                    frac2car_y_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_b_device), 
                    frac2car_z_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_c_device));

        if (minimum_distance_device[i] < 
            (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1))
        {
            minimum_distance_device[i] = (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1);
        }

        if (minimum_distance_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0])

            + cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_device[i]], charge_frame_device[index_frame_device[i]], 
                damping_a_device[0], minimum_distance_device[i]) 
            - cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_device[i]], charge_frame_device[index_frame_device[i]], 
                damping_a_device[0], cutoff_device[0]);
        }
    }
}









__global__
void s1_frac2cart_ini(int *num_inidividual_ini_extra_device, 

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device,
    double *ini_minimum_string_cart_x_device, double *ini_minimum_string_cart_y_device, double *ini_minimum_string_cart_z_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<((num_inidividual_ini_extra_device[0]+2)*3); i+=stride)
    {
        switch (i%3)
        {
            case 0:
                ini_minimum_string_cart_x_device[(i/3)] = 
                frac2car_x_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_a_device);
                break;
            case 1:
                ini_minimum_string_cart_y_device[(i/3)] = 
                frac2car_y_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_b_device);
                break;
            case 2:
                ini_minimum_string_cart_z_device[(i/3)] = 
                frac2car_z_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_c_device);
                break;
        }
    }
}



__global__
void cal_length_prep_ini(int *num_inidividual_ini_extra_device, 

    double *ini_minimum_string_cart_x_device, double *ini_minimum_string_cart_y_device, double *ini_minimum_string_cart_z_device,
    double *ini_minimum_length_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<((num_inidividual_ini_extra_device[0]+2)*3); i+=stride)
    {
        // for the first point
        if (((int) i/3)==0)
        {
            ini_minimum_length_all_device[i] = 0;
        }
        else
        {
            switch (((int) i%3))
            {
                case 0:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_x_device[(i/3)]-ini_minimum_string_cart_x_device[((i/3)-1)]),2);
                    break;
                case 1:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_y_device[(i/3)]-ini_minimum_string_cart_y_device[((i/3)-1)]),2);
                    break;
                case 2:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_z_device[(i/3)]-ini_minimum_string_cart_z_device[((i/3)-1)]),2);
                    break;
            }
        }

        

    }
}



__global__
void ini_length_sqrt_cal(int *num_inidividual_ini_extra_device, double *ini_minimum_length_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(num_inidividual_ini_extra_device[0]+2); i+=stride)
    {
        ini_minimum_length_device[i] = sqrt(ini_minimum_length_device[i]);
    }

}






__global__
void ini_2_s0(int *num_inidividual_ini_extra_device, int *N_string_device, 


    double *ini_minimum_l_abs_device, 

    double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    double *temp_partition_device, double *ini_minimum_length_device, 

    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        temp_partition_device[i] = 0;
        switch ((i%6))
        {
            case 0:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_a_device[(i/6)] = ini_minimum_string_a_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_a_device[ii+1]-ini_minimum_string_a_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }

                break;
            case 1:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_b_device[(i/6)] = ini_minimum_string_b_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_b_device[ii+1]-ini_minimum_string_b_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 2:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_c_device[(i/6)] = ini_minimum_string_c_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_c_device[ii+1]-ini_minimum_string_c_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 3:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_alpha_device[(i/6)] = ini_minimum_string_alpha_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_alpha_device[ii+1]-ini_minimum_string_alpha_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 4:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_beta_device[(i/6)] = ini_minimum_string_beta_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_beta_device[ii+1]-ini_minimum_string_beta_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 5:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_gamma_device[(i/6)] = ini_minimum_string_gamma_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_gamma_device[ii+1]-ini_minimum_string_gamma_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;

        }
    }
}



__global__
void Vext_cal_s0(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device, 
                double *charge_adsorbate_device, 
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *charge_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device, double *damping_a_device, 
                int *temp_add_frame_device,


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                int *N_string_device,

                double *s0_a_device, double *s0_b_device, double *s0_c_device, 
                double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,


                int *index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                int *index_adsorbate_cal_Vext_s0_device, int *index_frame_cal_Vext_s0_device,

                double *a_cal_Vext_s0_device, double *b_cal_Vext_s0_device, double *c_cal_Vext_s0_device,
                double *alpha_rad_cal_Vext_s0_device, double *beta_rad_cal_Vext_s0_device, double *gamma_rad_cal_Vext_s0_device,
                double *loc_x_cal_Vext_s0_device, double *loc_y_cal_Vext_s0_device, double *loc_z_cal_Vext_s0_device,
                double *vector_adsorbate_x_rot_cal_Vext_s0_device, double *vector_adsorbate_y_rot_cal_Vext_s0_device, double *vector_adsorbate_z_rot_cal_Vext_s0_device,
                double *adsorbate_cart_x_rot_cal_Vext_s0_device, double *adsorbate_cart_y_rot_cal_Vext_s0_device, double *adsorbate_cart_z_rot_cal_Vext_s0_device, 
                double *modify_frame_a_cal_Vext_s0_device, double *modify_frame_b_cal_Vext_s0_device, double *modify_frame_c_cal_Vext_s0_device,
                double *minimum_distance_cal_Vext_s0_device,
                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]); i+=stride)
    {
        index_s0_cal_Vext_s0_device[i] = (int) ( (i)/(N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        index_adsorbate_cal_Vext_s0_device[i] = (int) ( (i-index_s0_cal_Vext_s0_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(temp_add_frame_device[0]) );
        index_frame_cal_Vext_s0_device[i] = (int) ( (i - index_s0_cal_Vext_s0_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -index_adsorbate_cal_Vext_s0_device[i]*temp_add_frame_device[0]) );

        a_cal_Vext_s0_device[i] = s0_a_device[index_s0_cal_Vext_s0_device[i]];
        b_cal_Vext_s0_device[i] = s0_b_device[index_s0_cal_Vext_s0_device[i]];
        c_cal_Vext_s0_device[i] = s0_c_device[index_s0_cal_Vext_s0_device[i]];

        alpha_rad_cal_Vext_s0_device[i] = s0_alpha_device[index_s0_cal_Vext_s0_device[i]];
        beta_rad_cal_Vext_s0_device[i] = s0_beta_device[index_s0_cal_Vext_s0_device[i]];
        gamma_rad_cal_Vext_s0_device[i] = s0_gamma_device[index_s0_cal_Vext_s0_device[i]];


        loc_x_cal_Vext_s0_device[i] = frac2car_x_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_a_device);
        loc_y_cal_Vext_s0_device[i] = frac2car_y_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_b_device);
        loc_z_cal_Vext_s0_device[i] = frac2car_z_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_c_device);


        vector_adsorbate_x_rot_cal_Vext_s0_device[i] = rotate_moleucle_x_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);
        vector_adsorbate_y_rot_cal_Vext_s0_device[i] = rotate_moleucle_y_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);
        vector_adsorbate_z_rot_cal_Vext_s0_device[i] = rotate_moleucle_z_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);

        adsorbate_cart_x_rot_cal_Vext_s0_device[i] = loc_x_cal_Vext_s0_device[i]+vector_adsorbate_x_rot_cal_Vext_s0_device[i];
        adsorbate_cart_y_rot_cal_Vext_s0_device[i] = loc_y_cal_Vext_s0_device[i]+vector_adsorbate_y_rot_cal_Vext_s0_device[i];
        adsorbate_cart_z_rot_cal_Vext_s0_device[i] = loc_z_cal_Vext_s0_device[i]+vector_adsorbate_z_rot_cal_Vext_s0_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (adsorbate_cart_z_rot_cal_Vext_s0_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]] + times_z_device[0];
        }
        else if ( (adsorbate_cart_z_rot_cal_Vext_s0_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]] - times_z_device[0];
        }
        else
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]];
        }



        // y-direction
        if ( (adsorbate_cart_y_rot_cal_Vext_s0_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]] + times_y_device[0];
        }
        else if ( (adsorbate_cart_y_rot_cal_Vext_s0_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]] - times_y_device[0];
        }
        else
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]];
        }

        // x-direction
        if ( (adsorbate_cart_x_rot_cal_Vext_s0_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]] + times_x_device[0];
        }
        else if ( (adsorbate_cart_x_rot_cal_Vext_s0_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]] - times_x_device[0];
        }
        else
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]];
        }

        minimum_distance_cal_Vext_s0_device[i] = cal_dis_device(adsorbate_cart_x_rot_cal_Vext_s0_device[i], adsorbate_cart_y_rot_cal_Vext_s0_device[i], adsorbate_cart_z_rot_cal_Vext_s0_device[i], 
                    frac2car_x_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_a_device), 
                    frac2car_y_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_b_device), 
                    frac2car_z_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_c_device));

        if (minimum_distance_cal_Vext_s0_device[i] < 
            (((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5)*0.1))
        {
           minimum_distance_cal_Vext_s0_device[i] = (((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5)*0.1);
        }

        if (minimum_distance_cal_Vext_s0_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]*epsilon_frame_device[index_frame_cal_Vext_s0_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5), minimum_distance_cal_Vext_s0_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]*epsilon_frame_device[index_frame_cal_Vext_s0_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5), cutoff_device[0])


            + cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]], charge_frame_device[index_frame_cal_Vext_s0_device[i]], 
                damping_a_device[0], minimum_distance_cal_Vext_s0_device[i]) 
            - cal_pure_Coul_device(charge_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]], charge_frame_device[index_frame_cal_Vext_s0_device[i]], 
                damping_a_device[0], cutoff_device[0]);
        }
        
    }

}









__global__
void remap_string_var(int *N_atom_adsorbate_device, int *temp_add_frame_device,

                int *N_string_device,

                double *s0_a_device, double *s0_b_device, double *s0_c_device,
                double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,



                double *s0_deri_a_device, double *s0_deri_b_device, double *s0_deri_c_device, 
                double *s0_deri_alpha_device, double *s0_deri_beta_device, double *s0_deri_gamma_device,

                int *s0_deri_index_string_device, int *s0_deri_index_var_device,
                int *s0_deri_index_adsorbate_device, int *s0_deri_index_frame_device,


                double *move_angle_rad_device, double *move_frac_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]*7); i+=stride)
    {
        s0_deri_index_string_device[i] = (int) ( (i)/(7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        s0_deri_index_var_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        s0_deri_index_adsorbate_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_var_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(temp_add_frame_device[0]) );
        s0_deri_index_frame_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_var_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_adsorbate_device[i]*temp_add_frame_device[0]) );

        s0_deri_a_device[i] = s0_a_device[s0_deri_index_string_device[i]];
        s0_deri_b_device[i] = s0_b_device[s0_deri_index_string_device[i]];
        s0_deri_c_device[i] = s0_c_device[s0_deri_index_string_device[i]];
        s0_deri_alpha_device[i] = s0_alpha_device[s0_deri_index_string_device[i]];
        s0_deri_beta_device[i] = s0_beta_device[s0_deri_index_string_device[i]];
        s0_deri_gamma_device[i] = s0_gamma_device[s0_deri_index_string_device[i]];

        // modify the variable as the derivative requests
        switch (s0_deri_index_var_device[i])
        {
            case 1:
                s0_deri_a_device[i] = s0_deri_a_device[i] + move_frac_device[0];
                break;
            case 2:
                s0_deri_b_device[i] = s0_deri_b_device[i] + move_frac_device[0];
                break;
            case 3:
                s0_deri_c_device[i] = s0_deri_c_device[i] + move_frac_device[0];
                break;
            case 4:
                s0_deri_alpha_device[i] = s0_deri_alpha_device[i] + move_angle_rad_device[0];
                break;
            case 5:
                s0_deri_beta_device[i] = s0_deri_beta_device[i] + move_angle_rad_device[0];
                break;
            case 6:
                s0_deri_gamma_device[i] = s0_deri_gamma_device[i] + move_angle_rad_device[0];
                break;

        }
    }

}









__global__
void Vext_s0_deri_cal(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device, 
                double *charge_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *charge_frame_device,
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device, double *damping_a_device,
                int *temp_add_frame_device,

                int *N_string_device,

                double *s0_deri_a_device, double *s0_deri_b_device, double *s0_deri_c_device,
                double *s0_deri_alpha_device, double *s0_deri_beta_device, double *s0_deri_gamma_device,

                int *s0_deri_index_adsorbate_device, int *s0_deri_index_frame_device,

                double *s0_deri_loc_x_device, double *s0_deri_loc_y_device, double *s0_deri_loc_z_device,
                double *s0_deri_vector_adsorbate_x_rot_device, double *s0_deri_vector_adsorbate_y_rot_device, double *s0_deri_vector_adsorbate_z_rot_device,
                double *s0_deri_adsorbate_cart_x_rot_device, double *s0_deri_adsorbate_cart_y_rot_device, double *s0_deri_adsorbate_cart_z_rot_device,
                double *s0_deri_modify_frame_a_device, double *s0_deri_modify_frame_b_device, double *s0_deri_modify_frame_c_device,
                double *s0_deri_minimum_distance_device,

                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]*7); i+=stride)
    {
        s0_deri_loc_x_device[i] = frac2car_x_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_a_device);
        s0_deri_loc_y_device[i] = frac2car_y_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_b_device);
        s0_deri_loc_z_device[i] = frac2car_z_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_c_device);



        s0_deri_vector_adsorbate_x_rot_device[i] = rotate_moleucle_x_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);
        s0_deri_vector_adsorbate_y_rot_device[i] = rotate_moleucle_y_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);
        s0_deri_vector_adsorbate_z_rot_device[i] = rotate_moleucle_z_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);

        s0_deri_adsorbate_cart_x_rot_device[i] = s0_deri_loc_x_device[i]+s0_deri_vector_adsorbate_x_rot_device[i];
        s0_deri_adsorbate_cart_y_rot_device[i] = s0_deri_loc_y_device[i]+s0_deri_vector_adsorbate_y_rot_device[i];
        s0_deri_adsorbate_cart_z_rot_device[i] = s0_deri_loc_z_device[i]+s0_deri_vector_adsorbate_z_rot_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (s0_deri_adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]] + times_z_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]] - times_z_device[0];
        }
        else
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]];
        }



        // y-direction
        if ( (s0_deri_adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]] + times_y_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]] - times_y_device[0];
        }
        else
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]];
        }



        // x-direction
        if ( (s0_deri_adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]] + times_x_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]] - times_x_device[0];
        }
        else
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]];
        }



        s0_deri_minimum_distance_device[i] = cal_dis_device(s0_deri_adsorbate_cart_x_rot_device[i], s0_deri_adsorbate_cart_y_rot_device[i], s0_deri_adsorbate_cart_z_rot_device[i], 
                    frac2car_x_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_a_device), 
                    frac2car_y_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_b_device), 
                    frac2car_z_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_c_device));



        if (s0_deri_minimum_distance_device[i] < 
            (((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5)*0.1))
        {
           s0_deri_minimum_distance_device[i] = (((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5)*0.1);
        }

        if (s0_deri_minimum_distance_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[s0_deri_index_adsorbate_device[i]]*epsilon_frame_device[s0_deri_index_frame_device[i]]), 
                ((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5), s0_deri_minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[s0_deri_index_adsorbate_device[i]]*epsilon_frame_device[s0_deri_index_frame_device[i]]), 
                ((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5), cutoff_device[0])


            + cal_pure_Coul_device(charge_adsorbate_device[s0_deri_index_adsorbate_device[i]], charge_frame_device[s0_deri_index_frame_device[i]], 
                damping_a_device[0], s0_deri_minimum_distance_device[i]) 
            - cal_pure_Coul_device(charge_adsorbate_device[s0_deri_index_adsorbate_device[i]], charge_frame_device[s0_deri_index_frame_device[i]], 
                damping_a_device[0], cutoff_device[0]);
        }
        
    }

}

__global__
void s0_grad_cal(double *move_frac_device, double *move_angle_rad_device, double *rounding_coeff_device,
    int *N_string_device, double *s0_deri_Vext_device, double *s0_gradient_device, double *s0_gradient_square_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {


        s0_gradient_device[i] = s0_deri_Vext_device[(((int) (i%6))+1 + ((int) (i/6))*7)]-s0_deri_Vext_device[((int) (i/6))*7];
        
        switch (((int) (i%6)))
        {
            case 0:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 1:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 2:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 3:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 4:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 5:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
        }
        
        s0_gradient_device[i] = floor(s0_gradient_device[i]/rounding_coeff_device[0])*rounding_coeff_device[0];

        s0_gradient_square_device[i] = s0_gradient_device[i]*s0_gradient_device[i];

    }

}



__global__
void s0_grad_length_sqrt_cal(double *rounding_coeff_device, int *N_string_device, double *s0_gradient_length_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        s0_gradient_length_device[i] = floor(sqrt(s0_gradient_length_device[i])/rounding_coeff_device[0])*rounding_coeff_device[0];
    }

}





__global__
void s0_new_cal(double *rounding_coeff_device, int *N_string_device, 
    double *move_frac_device, double *move_angle_rad_device,




    double *s0_gradient_length_device, double *s0_gradient_device,
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,
    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;


    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        switch (i%6)
        {
            case 0:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_a_device[((int) (i/6))] = s0_a_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_a_device[((int) (i/6))] = s0_a_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 1:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_b_device[((int) (i/6))] = s0_b_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_b_device[((int) (i/6))] = s0_b_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 2:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_c_device[((int) (i/6))] = s0_c_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_c_device[((int) (i/6))] = s0_c_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;

            case 3:
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 4:
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 5:
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];

                }
                break;
        }

    }

}












__global__
void s1_fix_modify_upgrade(int *N_string_device, 

    int *direction_device,

    double *s0_gradient_length_device, double *s0_gradient_device,
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,
    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_a_device[0] = s0_a_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_a_device[N_string_device[0]-1] = s0_a_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_b_device[N_string_device[0]-1] = s1_b_device[0];
                    s1_c_device[N_string_device[0]-1] = s1_c_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_a_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_a_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_a_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_a_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }
                break;

            case 2:

                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_b_device[0] = s0_b_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_b_device[N_string_device[0]-1] = s0_b_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_a_device[N_string_device[0]-1] = s1_a_device[0];
                    s1_c_device[N_string_device[0]-1] = s1_c_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_b_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_b_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_b_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_b_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }

                break;


            case 3:

                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_c_device[0] = s0_c_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_c_device[N_string_device[0]-1] = s0_c_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_a_device[N_string_device[0]-1] = s1_a_device[0];
                    s1_b_device[N_string_device[0]-1] = s1_b_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_c_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_c_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_c_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_c_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }

                break;
        }
        
    }
}





__global__
void s1_frac2cart(int *N_string_device, 

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_cart_x_device, double *s1_cart_y_device, double *s1_cart_z_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]); i+=stride)
    {
        s1_cart_x_device[i] = frac2car_x_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_a_device);
        s1_cart_y_device[i] = frac2car_y_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_b_device);
        s1_cart_z_device[i] = frac2car_z_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_c_device);

    }
}




__global__
void s1_length_prep(int *N_string_device, 

    double *s1_cart_x_device, double *s1_cart_y_device, double *s1_cart_z_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device,
    double *s1_length_coordinate_all_device, double *s1_length_orientation_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        // for the first point
        if (((int) i/6)==0)
        {
            s1_length_coordinate_all_device[(i/6)*3+(i%6)] = 0;

            s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = 0;
        }
        else
        {
            switch (((int) i%6))
            {
                case 0:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_x_device[((int) i/6)]-s1_cart_x_device[((int) i/6)-1]),2);
                    break;
                case 1:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_y_device[((int) i/6)]-s1_cart_y_device[((int) i/6)-1]),2);
                    break;
                case 2:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_z_device[((int) i/6)]-s1_cart_z_device[((int) i/6)-1]),2);
                    break;

                case 3:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_alpha_device[((int) i/6)] -s1_alpha_device[((int) i/6)-1]),2);
                    break;
                case 4:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_beta_device[((int) i/6)] -s1_beta_device[((int) i/6)-1]),2);
                    break;
                case 5:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_gamma_device[((int) i/6)] -s1_gamma_device[((int) i/6)-1]),2);
                    break;
            }
        }

        

    }
}



__global__
void s1_length_sqrt_cal(double *rounding_coeff_device, int *N_string_device, 
    double *s1_length_coordinate_device, double *s1_length_orientation_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                s1_length_coordinate_device[(i/2)] = sqrt(s1_length_coordinate_device[(i/2)]);
                break;
            case 1:
                s1_length_orientation_device[(i/2)] = sqrt(s1_length_orientation_device[(i/2)]);
                break;
        }
    }

}

__global__
void remap_s1_length_for_cumulation(int *N_string_device, double *s1_length_coordinate_device, double *s1_length_orientation_device,

    double *s1_length_coordinate_remap_device, double *s1_length_orientation_remap_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<((int) (N_string_device[0]*(1+N_string_device[0])*0.5*2)); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                s1_length_coordinate_remap_device[(i/2)] = 
                s1_length_coordinate_device[((int) ((i/2) - ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1)*(1+ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1))*0.5))];
                break;
            case 1:
                s1_length_orientation_remap_device[(i/2)] = 
                s1_length_orientation_device[((int) ((i/2) - ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1)*(1+ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1))*0.5))];
                break;
        }
    }

}



__global__
void s1_2_s2(int *N_string_device, double *s1_l_abs_coordinate_device, double *s1_l_abs_orientation_device,
    double *s1_length_coordinate_cumulation_device, double *s1_length_orientation_cumulation_device,

    double *s1_a_device, double *s1_b_device, double *s1_c_device, 
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device,

    double *s2_a_device, double *s2_b_device, double *s2_c_device,
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                for (ii=(i/2); ;)
                {
                    if(ii==(N_string_device[0]-1))
                    {
                        ii--;
                    }

                    // go beyond the lower boundary
                    if ((1.0*(i/2)/(N_string_device[0]-1)) < (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0]))
                    {
                        ii--;
                    }
                    // go beyond the upper boundary
                    else if ((1.0*(i/2)/(N_string_device[0]-1)) > (1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]))
                    {
                        ii++;
                    }
                    else if ( ((1.0*(i/2)/(N_string_device[0]-1)) >= (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0]))
                        && ((1.0*(i/2)/(N_string_device[0]-1)) <= (1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0])) )
                    {
                        break;
                    }
                }
                s2_a_device[(i/2)] = s1_a_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_a_device[ii+1]-s1_a_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                s2_b_device[(i/2)] = s1_b_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_b_device[ii+1]-s1_b_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                s2_c_device[(i/2)] = s1_c_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_c_device[ii+1]-s1_c_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                break;

            case 1:
                    s2_alpha_device[(i/2)] = s1_alpha_device[(i/2)];
                    s2_beta_device[(i/2)] = s1_beta_device[(i/2)];
                    s2_gamma_device[(i/2)] = s1_gamma_device[(i/2)];
                break;
        }  
    }
}







__global__
void check_s2(int *N_string_device, double *V_s0, double *V_s2,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device, 
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*3); i+=stride)
    {
        switch ((int) (i%3))
        {
            case 0:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_alpha_device[((int) (i/3))] = s0_alpha_device[((int) (i/3))];
                }
                break;

            case 1:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_beta_device[((int) (i/3))] = s0_beta_device[((int) (i/3))];
                }
                break;

            case 2:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_gamma_device[((int) (i/3))] = s0_gamma_device[((int) (i/3))];
                }
                break;
        }  
    }
}








__global__
void smooth_angle(int *N_string_device, double *smooth_coeff_device, 
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device, 
    double *s2_alpha_smooth_device, double *s2_beta_smooth_device, double *s2_gamma_smooth_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*3); i+=stride)
    {
        switch ((int) (i%3))
        {
            case 0:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_alpha_smooth_device[(i/3)] = s2_alpha_device[(i/3)];
                }
                else
                {
                    if ((s2_alpha_device[(i/3)-1]-s2_alpha_device[(i/3)]) > PI)
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) > PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_alpha_device[(i/3)-1]-s2_alpha_device[(i/3)]) < -PI)
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) < -PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) > PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) < -PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]);
                        }
                    }
                }
                break;

            case 1:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_beta_smooth_device[(i/3)] = s2_beta_device[(i/3)];
                }
                else
                {
                    if ((s2_beta_device[(i/3)-1]-s2_beta_device[(i/3)]) > PI)
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) > PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_beta_device[(i/3)-1]-s2_beta_device[(i/3)]) < -PI)
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) < -PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) > PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) < -PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]);
                        }
                    }
                }
                break;

            case 2:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_gamma_smooth_device[(i/3)] = s2_gamma_device[(i/3)];
                }
                else
                {
                    if ((s2_gamma_device[(i/3)-1]-s2_gamma_device[(i/3)]) > PI)
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) > PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_gamma_device[(i/3)-1]-s2_gamma_device[(i/3)]) < -PI)
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) < -PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) > PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) < -PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]);
                        }
                    }
                }
                break;
        }  
    }
}





__global__
void diff_s_prep(int *N_string_device, 

    double *s0_a_device, double *s0_b_device, double *s0_c_device,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,

    double *s_final_a_device, double *s_final_b_device, double *s_final_c_device,
    double *s_final_alpha_device, double *s_final_beta_device, double *s_final_gamma_device,

    double *diff_s_coordinate_all_device, double *diff_s_orientation_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        
        switch (((int) i%6))
        {
            case 0:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_a_device[((int) i/6)]-s_final_a_device[((int) i/6)]),2);
                break;
            case 1:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_b_device[((int) i/6)]-s_final_b_device[((int) i/6)]),2);
                break;
            case 2:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_c_device[((int) i/6)]-s_final_c_device[((int) i/6)]),2);
                break;
            case 3:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_alpha_device[((int) i/6)]-s_final_alpha_device[((int) i/6)]),2);
                break;
            case 4:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_beta_device[((int) i/6)]-s_final_beta_device[((int) i/6)]),2);
                break;
            case 5:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_gamma_device[((int) i/6)]-s_final_gamma_device[((int) i/6)]),2);
                break;
        }

        

    }
}



__global__
void check_signal(int *N_string_device, 
    double *total_diff_s_coordinate_device, double *total_diff_s_orientation_device,
    double *convergence_coorindate_device, double *convergence_orientation_device,
    int *signal_coordinate_device, int *signal_orientation_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<2; i+=stride)
    {
        switch (i)
        {
            case 0:
                if (signal_coordinate_device[0] == 0)
                {
                    if (total_diff_s_coordinate_device[0] < convergence_coorindate_device[0])
                    {
                        // if (total_diff_s_orientation_device[0] < convergence_orientation_device[0])
                        // {
                            signal_coordinate_device[0] = 1;
                        // }
                    }
                }
                break;
            case 1:
                if (signal_orientation_device[0] == 0)
                {
                    if (total_diff_s_orientation_device[0] < convergence_orientation_device[0])
                    {
                        // if (total_diff_s_coordinate_device[0] < convergence_coorindate_device[0])
                        // {
                            signal_orientation_device[0] = 1;
                        // }
                    }
                }
                break;
        }
    }
}





__global__
void copy2s0(int *N_string_device, 
    int *signal_coordinate_device, int *signal_orientation_device,

    double *s_copy_a_device, double *s_copy_b_device, double *s_copy_c_device, 
    double *s_copy_alpha_device, double *s_copy_beta_device, double *s_copy_gamma_device, 
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        switch ((int) (i%6))
        {
            case 0:
                    s0_a_device[((int) (i/6))] = s_copy_a_device[((int) (i/6))];
                break;
            case 1:
                    s0_b_device[((int) (i/6))] = s_copy_b_device[((int) (i/6))];
                break;
            case 2:
                    s0_c_device[((int) (i/6))] = s_copy_c_device[((int) (i/6))];
                break;
            case 3:
                if (s_copy_alpha_device[((int) (i/6))] > (2*PI))
                {
                    s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))] - 2*PI;
                }
                else if (s_copy_alpha_device[((int) (i/6))] < 0)
                {
                    s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))] + 2*PI;
                }
                else
                {
                    s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))];
                }
                break;
            case 4:
                if (s_copy_beta_device[((int) (i/6))] > (2*PI))
                {
                    s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))] - 2*PI;
                }
                else if (s_copy_beta_device[((int) (i/6))] < 0)
                {
                    s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))] + 2*PI;
                }
                else
                {
                    s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))];
                }
                break;
            case 5:
                if (s_copy_gamma_device[((int) (i/6))] > (2*PI))
                {
                    s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))] - 2*PI;
                }
                else if (s_copy_gamma_device[((int) (i/6))] < 0)
                {
                    s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))] + 2*PI;
                }
                else
                {
                    s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))];
                }
                break;
        }  
    }
}















