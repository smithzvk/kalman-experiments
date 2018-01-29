#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "kalman.h"

kf_t *
kf_alloc(size_t n, size_t m)
{
   kf_t *kf = (kf_t *) malloc(sizeof(kf_t));

   kf->F = gsl_matrix_alloc(n, n);
   gsl_matrix_set_identity(kf->F);

   kf->H = gsl_matrix_calloc(m, n);
   kf->Q = gsl_matrix_calloc(n, n);

   kf->R = gsl_matrix_alloc(m, m);
   /* R must not be a zero matrix, but the identity matrix is possibly a very
    * large error */
   gsl_matrix_set_identity(kf->R);

   kf->x = gsl_vector_calloc(n);
   kf->Fx = gsl_vector_calloc(n);
   kf->Hx = gsl_vector_calloc(m);

   kf->P = gsl_matrix_calloc(n, n);
   kf->Pp = gsl_matrix_calloc(n, n);

   kf->K = gsl_matrix_alloc(n, m);

   kf->S_valid = 0;
   kf->S = gsl_matrix_alloc(m, m);
   kf->S_low = gsl_matrix_alloc(m, m);
   kf->S_inv = gsl_matrix_alloc(m, m);
   kf->Sdet = 1.0;

   kf->y = gsl_vector_alloc(m);
   kf->Ky = gsl_vector_alloc(n);
   kf->KH = gsl_matrix_alloc(n, n);
   kf->FP = gsl_matrix_alloc(n, n);
   kf->PHt = gsl_matrix_alloc(n, m);

   kf->fadingMemoryAlphaSq = 1.0;
   kf->sigmaSq = 1.0;

   return kf;
}

kf_status
kf_init(kf_t *kf)
{
   kf->S_valid = 0;
   kf->Sdet = 1.0;
   kf->fadingMemoryAlphaSq = 1.0;
   kf->sigmaSq = 1.0;
   return kf_success;
}

kf_status
kf_free(kf_t *kf)
{
   gsl_matrix_free(kf->F);
   gsl_matrix_free(kf->H);
   gsl_matrix_free(kf->R);
   gsl_matrix_free(kf->Q);

   gsl_vector_free(kf->x);
   gsl_vector_free(kf->Fx);
   gsl_vector_free(kf->Hx);
   gsl_matrix_free(kf->P);
   gsl_matrix_free(kf->Pp);

   gsl_matrix_free(kf->K);

   gsl_matrix_free(kf->S);
   gsl_matrix_free(kf->S_low);
   gsl_matrix_free(kf->S_inv);

   gsl_vector_free(kf->y);
   gsl_vector_free(kf->Ky);
   gsl_matrix_free(kf->KH);
   gsl_matrix_free(kf->FP);
   gsl_matrix_free(kf->PHt);

   free(kf);
   return kf_success;
}

kf_status
kf_predict(kf_t *kf)
{
   /* Compute Fx and Hx*/
   gsl_blas_dgemv(CblasNoTrans, 1.0, kf->F, kf->x, 0.0, kf->Fx);
   gsl_blas_dgemv(CblasNoTrans, 1.0, kf->H, kf->x, 0.0, kf->Hx);

   /* Compute Pp */
   gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->F, kf->P, 0.0, kf->FP);
   gsl_matrix_memcpy(kf->Pp, kf->Q);
   gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                  kf->fadingMemoryAlphaSq, kf->FP, kf->F,
                  kf->sigmaSq, kf->Pp);

   kf->S_valid = 0;
   return kf_success;
}

kf_status
kf_eval(kf_t *kf, gsl_vector *z, gsl_matrix *R, KF_FLOAT *ll)
{
   gsl_matrix_memcpy(kf->S, R);
   gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, kf->Pp, kf->H, 1.0, kf->PHt);
   gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->H, kf->PHt, 1.0, kf->S);

   gsl_vector_memcpy(kf->y, z);
   gsl_blas_dgemv(CblasNoTrans, -1.0, kf->H, kf->Fx, 1.0, kf->y);

   gsl_matrix_memcpy(kf->S_low, kf->S);
   gsl_linalg_cholesky_decomp1(kf->S_low);

   double det = 1;
   for (int i = 0; i < kf->S_low->size1; i++)
      det *= gsl_matrix_get(kf->S_low, i, i);
   kf->Sdet = pow(det, 2);

   gsl_matrix_memcpy(kf->S_inv, kf->S_low);
   gsl_linalg_cholesky_invert(kf->S_inv);

   kf->S_valid = 1;

   return kf_success;
}

kf_status
kf_update(kf_t *kf, gsl_vector *z, gsl_matrix *R)
{
   if (R == NULL)
      R = kf->R;

   KF_FLOAT ll;
   /* If we don't pass a new measurement, then just use the previous kf_eval
    * result. */
   if (z != NULL)
      kf_eval(kf, z, R, &ll);

   if (!kf->S_valid)
      return kf_failure;

   /* K = P H^T S^-1 */
   gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->PHt, kf->S_inv, 0, kf->K);

   /* x' = F x + K y */
   gsl_vector_memcpy(kf->x, kf->Fx);
   gsl_blas_dgemv(CblasNoTrans, 1.0, kf->K, kf->y, 1.0, kf->x);
   /* gsl_blas_daxpy(1.0, kf->Ky, kf->x); */

   /* P' = Pp + Q - K H Pp */
   gsl_matrix_memcpy(kf->P, kf->Pp);
   gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kf->K, kf->H, 0.0, kf->KH);
   gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, kf->KH, kf->Pp, 1.0, kf->P);

   return kf_success;
}
