#include "kalman.h"
#include <math.h>

#define SIMPLE_NOISE_MODEL 0
#define CONTINUOUS_NOISE_MODEL 1
#define PIECEWISE_NOISE_MODEL 2

kf_status
setProcessNoise(gsl_matrix *Q, int noiseModel, KF_FLOAT dt)
{
   if (SIMPLE_NOISE_MODEL == noiseModel)
   {
      gsl_matrix_set(Q, 0, 0, 0.0); gsl_matrix_set(Q, 0, 1, 0.0);
      gsl_matrix_set(Q, 1, 0, 0.0); gsl_matrix_set(Q, 1, 1, dt);
   }
   else if (CONTINUOUS_NOISE_MODEL == noiseModel)
   {
      double dt2 = dt*dt;
      double dt3 = dt2*dt;
      gsl_matrix_set(Q, 0, 0, dt3/3.0); gsl_matrix_set(Q, 0, 1, dt2/2.0);
      gsl_matrix_set(Q, 1, 0, dt2/2.0); gsl_matrix_set(Q, 1, 1, dt);
   }
   else if (PIECEWISE_NOISE_MODEL == noiseModel)
   {
      double dt2 = dt*dt;
      double dt3 = dt2*dt;
      double dt4 = dt2*dt2;
      gsl_matrix_set(Q, 0, 0, dt4/4.0); gsl_matrix_set(Q, 0, 1, dt3/2.0);
      gsl_matrix_set(Q, 1, 0, dt3/2.0); gsl_matrix_set(Q, 1, 1, dt2);
   }
   return kf_success;
}

#define N_POINTS 50

/* Parabola with added noise */
double parabola[N_POINTS] = {0.0485694034653, 0.0548351524804, 0.0468714337096,
                             0.0908087881534, 0.051225640846, 0.0708488560165,
                             0.146347751812, 0.140626529574, 0.134085408885,
                             0.110884293681, 0.19610569709, 0.183388635803,
                             0.148803241238, 0.217864356027, 0.222439529756,
                             0.209136150549, 0.198903389241, 0.194575938483,
                             0.197107383272, 0.204531935394, 0.196555787681,
                             0.201144475741, 0.280319559433, 0.236979888452,
                             0.269754762039, 0.264708451603, 0.214612639898,
                             0.248609772499, 0.241965948258, 0.212717299595,
                             0.204831656858, 0.220083048679, 0.206838229965,
                             0.255738430021, 0.191641359558, 0.20726122237,
                             0.17845543904, 0.134668298598, 0.15157034733,
                             0.151581453466, 0.136233367302, 0.14290164864,
                             0.145580395438, 0.062240665641, 0.0772342338583,
                             0.117692816251, 0.0437538411719, 0.0877472884267,
                             5.8772802522e-3, 7.1591578256e-3};

/* Constant velocity with added noise */
double constantVel[N_POINTS] = {4.1197562567e-3, 0.0425693863809,
                                0.068300714589, 0.0405101295827, 0.126443441742,
                                0.146305267313, 0.123678691453, 0.174932688492,
                                0.185480330659, 0.242586125084, 0.187496446031,
                                0.200026736477, 0.231323231169, 0.296367148827,
                                0.346741438277, 0.331459458843, 0.362953102013,
                                0.358752638268, 0.398501193212, 0.37266661233,
                                0.388564058792, 0.447522020435, 0.415011179514,
                                0.443361055209, 0.502046547104, 0.567444528685,
                                0.543432907723, 0.589687950694, 0.629186849757,
                                0.564720426473, 0.61129592866, 0.687679641741,
                                0.674542600655, 0.697367821044, 0.739385394814,
                                0.749901030749, 0.716634130263, 0.738790240739,
                                0.821427529362, 0.796846728212, 0.851898340934,
                                0.855480993681, 0.900346106844, 0.925431927901,
                                0.859363087941, 0.880360337071, 0.890950783338,
                                0.963543040113, 0.944225479693, 0.98403617954};

int
main()
{
   int n = 2;
   int m = 1;

   double dt = .01;

   double F[] = {1.0, dt,
                 0.0, 1.0};
   double H[] = {1.0, 0};

   double R[] = {.01};

   kf_t *kf = kf_alloc(n, m);
   kf_init(kf);

   kf->sigmaSq = 0.001;

   gsl_vector_set(kf->x, 0, 0);
   gsl_vector_set(kf->x, 1, 0);

   gsl_matrix_set(kf->P, 0, 0, 1.0); gsl_matrix_set(kf->P, 0, 1, 0.0);
   gsl_matrix_set(kf->P, 1, 0, 0.0); gsl_matrix_set(kf->P, 1, 1, 100.0);

   gsl_matrix_view Rmat = gsl_matrix_view_array(R, 1, 1);
   gsl_matrix_memcpy(kf->R, &Rmat.matrix);
   gsl_matrix_view Hmat = gsl_matrix_view_array(H, 1, 2);
   gsl_matrix_memcpy(kf->H, &Hmat.matrix);
   gsl_matrix_view Fmat = gsl_matrix_view_array(F, 2, 2);
   gsl_matrix_memcpy(kf->F, &Fmat.matrix);

   setProcessNoise(kf->Q, PIECEWISE_NOISE_MODEL, 0.1);

   for (int i = 0; i < N_POINTS; i++)
   {
      printf("Filtered State: %f +/- %f, %f +/- %f\n",
             gsl_vector_get(kf->x, 0),
             sqrt(gsl_matrix_get(kf->P, 0, 0)),
             gsl_vector_get(kf->x, 1),
             sqrt(gsl_matrix_get(kf->P, 1, 1)));

      kf_predict(kf);

      printf("Predicted State: %f +/- %f, %f +/- %f\n",
             gsl_vector_get(kf->Fx, 0),
             sqrt(gsl_matrix_get(kf->Pp, 0, 0)),
             gsl_vector_get(kf->Fx, 1),
             sqrt(gsl_matrix_get(kf->Pp, 1, 1)));

      gsl_vector_view z = gsl_vector_view_array(&constantVel[i], 1);
      printf("Measurement: %f\n", gsl_vector_get(&z.vector, 0));

      kf_update(kf, &z.vector, NULL);
   }

   kf_free(kf);

   return 0;
}
