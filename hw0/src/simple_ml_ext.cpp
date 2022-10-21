#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <memory.h>

namespace py = pybind11;

void compute_crossentropy_grad(const float* out, const unsigned char * label, float* grad, size_t batch, size_t classes){
    for(int i = 0; i < batch; i++){
        float accum = 0.0f; 
        for(int j = 0; j < classes; j++){
            accum += std::exp(out[i * classes + j]); 
        }
        float grad_val = 0.0f; 
        for(int j = 0; j < classes; j++){
            grad_val = std::exp(out[i * classes + j]) / accum; 
            if(j == label[i]){
                grad_val -= 1; 
            }
            // Reduce Mean
            grad_val /= batch; 
            grad[i * classes + j] = grad_val; 
        }
    }
}


void matrixTranspose(const float* a, float* res, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res[j * m + i] = a[i * n + j];
    }
  }
  return ;
} 

void matmul(const float* x, const float* w, float* y, const size_t m, const size_t n, const size_t k){
    /*
    x: m, n
    w: n, k
    y: m, k
    */
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            float accum = 0.0f; 
            for(int t = 0; t < n; t++){
                accum += x[i * n + t] * w[t * k + j]; 
            }
            y[i * k + j] = accum; 
        }
    }
}

void update(const float* grad, float* w, float lr, size_t elem_cnt){
    for(int i = 0; i < elem_cnt; i++){
        w[i] -= lr * grad[i]; 
    }
}




void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int sample_num = m; 
    int iter_num = (sample_num + (batch - 1)) / batch;
    
    float* theta_grad; 
    theta_grad = (float*)malloc(n*k*sizeof(float)); 
     

    for(int iter = 0; iter < iter_num; iter++){
        int start = iter * batch * n;
        int end = (iter + 1) * batch * n;
        end = end > m * n ? m * n : end;
        int num = (end - start) / n;

        const float* iter_x = X + start; 
        const unsigned char * iter_y = y + start;

        float* matmul_result; 
        matmul_result = (float*)malloc(num*k*sizeof(float)); 
        float* cross_entropy_grad; 
        cross_entropy_grad = (float*)malloc(num*k*sizeof(float));
        float* transposed_x; 
        transposed_x = (float*)malloc(n*num*sizeof(float)); 

        matmul(iter_x, theta, matmul_result, num, n, k); 
        compute_crossentropy_grad(matmul_result, iter_y, cross_entropy_grad, num, k); 
        matrixTranspose(iter_x, transposed_x, num, n);
        matmul(transposed_x, cross_entropy_grad, theta_grad, n, num, k); 
        update(theta_grad, theta, lr, n * k); 

        free(matmul_result); 
        free(cross_entropy_grad); 
        free(transposed_x);
    }

    free(theta_grad);
     
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
