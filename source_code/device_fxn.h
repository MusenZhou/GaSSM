#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32
#define coul2Klevin 1.6710095663e+05




__device__
double frac2car_x_device(double frac_a, double frac_b, double frac_c, double *frac2car_a_device)
{
    return (frac_a*frac2car_a_device[0] + frac_b*frac2car_a_device[1] + frac_c*frac2car_a_device[2]);
}

__device__
double frac2car_y_device(double frac_a, double frac_b, double frac_c, double *frac2car_b_device)
{
    return (frac_a*frac2car_b_device[0] + frac_b*frac2car_b_device[1] + frac_c*frac2car_b_device[2]);
}

__device__
double frac2car_z_device(double frac_a, double frac_b, double frac_c, double *frac2car_c_device)
{
    return (frac_a*frac2car_c_device[0] + frac_b*frac2car_c_device[1] + frac_c*frac2car_c_device[2]);
}




__device__
double rotate_moleucle_x_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( vector_adsorbate_x_device*cos(rot_gamma_rad)*cos(rot_beta_rad) 
            - vector_adsorbate_y_device*sin(rot_gamma_rad)*cos(rot_alpha_rad) 
            + vector_adsorbate_y_device*sin(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) 
            + vector_adsorbate_z_device*sin(rot_gamma_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*cos(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) );
}

__device__
double rotate_moleucle_y_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( vector_adsorbate_x_device*sin(rot_gamma_rad)*cos(rot_beta_rad) 
            + vector_adsorbate_y_device*cos(rot_gamma_rad)*cos(rot_alpha_rad) 
            + vector_adsorbate_y_device*sin(rot_gamma_rad)*sin(rot_beta_rad)*sin(rot_alpha_rad)
            - vector_adsorbate_z_device*cos(rot_gamma_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*sin(rot_gamma_rad)*sin(rot_beta_rad)*cos(rot_alpha_rad) );
}

__device__
double rotate_moleucle_z_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( -vector_adsorbate_x_device*sin(rot_beta_rad) 
            + vector_adsorbate_y_device*cos(rot_beta_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*cos(rot_alpha_rad)*cos(rot_beta_rad) );
}


__device__
double cal_dis_device(double loc_x_device, double loc_y_device, double loc_z_device, 
    double frame_x_device, double frame_y_device, double frame_z_device)
{

    return ( sqrt( pow((loc_x_device-frame_x_device),2)+pow((loc_y_device-frame_y_device),2)+pow((loc_z_device-frame_z_device),2) ) );
}



__device__
double cal_pure_lj_device(double epsilon_cal_device, double sigma_cal_device, double distance)
{
    return ( 4*epsilon_cal_device*(pow((sigma_cal_device/distance),12) - pow((sigma_cal_device/distance),6)) );
}



__device__
double cal_pure_Coul_device(double charge1_device, double charge2_device, double damping_a_device, double distance)
{
    return ( 1.0*coul2Klevin*charge1_device*charge2_device*erfc(damping_a_device*distance)/distance );
}
