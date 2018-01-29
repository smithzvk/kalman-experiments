
#ifndef __KF_KALMAN_H__
#define __KF_KALMAN_H__

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#define KF_FLOAT double

#define N_VECTOR(x) gsl_vector *(x)
#define M_VECTOR(x) gsl_vector *(x)

#define NN_MATRIX(x) gsl_matrix *(x)
#define MM_MATRIX(x) gsl_matrix *(x)

#define NM_MATRIX(x) gsl_matrix *(x)
#define MN_MATRIX(x) NM_MATRIX(x)

typedef struct
{
   /* Model definition */
   NN_MATRIX(F);                // Evolution matrix (x' = F x)
   MN_MATRIX(H);                // Measurement matrix (measurment = H x)
   MM_MATRIX(R);                // Measurement error matrix
   NN_MATRIX(Q);                // Process error matrix

   /* State data */
   N_VECTOR(x);                 // State vector
   N_VECTOR(Fx);                // Predicted new state vector
   M_VECTOR(Hx);                // Measurement corresponding to state x
   NN_MATRIX(P);                // Covariance matrix
   NN_MATRIX(Pp);               // Predicted new covariance matrix

   /* Kalman gain */
   NM_MATRIX(K);

   /* Innovation */
   int S_valid;
   MM_MATRIX(S);
   MM_MATRIX(S_low);
   MM_MATRIX(S_inv);
   KF_FLOAT Sdet;

   /* Internal memory */
   M_VECTOR(y);                 // Measurement residual
   M_VECTOR(Ky);                // K * y
   NN_MATRIX(KH);
   NN_MATRIX(FP);
   NM_MATRIX(PHt);

   /* Control parameters */
   KF_FLOAT fadingMemoryAlphaSq; // alpha >= 1.0. 1.0 = perfect memory
   KF_FLOAT sigmaSq; // Process error scale
} kf_t;

kf_t *
kf_alloc(size_t n, size_t m);

typedef enum
{
   kf_success = 0,
   kf_failure = -1
} kf_status;

kf_status
kf_init(kf_t *kf);

kf_status
kf_free(kf_t *kf);

kf_status
kf_predict(kf_t *kf);

kf_status
kf_eval(kf_t *kf, gsl_vector *z, gsl_matrix *R, KF_FLOAT *ll);

kf_status
kf_update(kf_t *kf, gsl_vector *z, gsl_matrix *R);

#undef N_VECTOR
#undef M_VECTOR

#undef NN_MATRIX
#undef MM_MATRIX

#undef NM_MATRIX
#undef MN_MATRIX

#endif
