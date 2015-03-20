#include <cfloat>
#include <vector>
#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>

/*

  Takes in the locations (x1,y1,x2,y2,...,xn,yn)
  and a mask (m1, m2, ..., mn)
  Produces output (x'1,y'1,x'2,y'2,...,x'n,y'n)
  where x'i = mi * xi, y'i = mi * yi
  bottom[0] -> locations
  bottom[1] -> mask
  top[0]    -> locations

*/

namespace caffe {

// template <typename Dtype>
// void LocMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {

//   // output as many values as input contains
//   top[0]->ReshapeLike(*bottom[0]);

//   // make sure locations has twice as many elements as mask
//   CHECK(bottom[0]->channels() == 2 * bottom[1]->channels());
// }

template <typename Dtype>
void LocMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();      // values coming from below
  const Dtype* bottom_mask = bottom[1]->cpu_data();
  Dtype* top_data          = top[0]->mutable_cpu_data(); // values to be sent above


  const vector<int> bottom_shape = bottom[0]->shape(); // N, C, H, W
  const vector<int> mask_shape = bottom[1]->shape(); // N, C, H, W


/*
  std::cout << "Mask: " ;
  for (int n = 0; n < mask_shape[0]; ++n) {
      for (int i = 0; i < mask_shape[1]; ++i) {
          std::cout << bottom_mask[n*mask_shape[1]+i] << ", ";
      }
      std::cout << "\n";
  }


  std::cout << "Bottom: " ;
  for (int n = 0; n < bottom_shape[0]; ++n) {
      for (int i = 0; i < bottom_shape[1]; ++i) {
          std::cout << bottom_data[n*bottom_shape[1]+i] << ", ";
      }
      std::cout << "\n";
  }
*/
  for (int n = 0; n < bottom_shape[0]; ++n) {
      for(int k = 0; k < mask_shape[1]; ++k){

          int idx_1 = (n*bottom_shape[1]+2*k);
          int idx_2 = (n*bottom_shape[1]+2*k+1);
          int mask_idx = n*mask_shape[1]+k;
          top_data[idx_1] = bottom_data[idx_1] *sqrt(bottom_mask[mask_idx]);
          top_data[idx_2] = bottom_data[idx_2] *sqrt(bottom_mask[mask_idx]);
          if(top_data[idx_1] < 1e-30)
              top_data[idx_1] = 0;
          if(top_data[idx_2] < 1e-30)
              top_data[idx_2] = 0;


      }
  }
  /*
  std::cout << "top: " ;
  for (int n = 0; n < bottom_shape[0]; ++n) {
      for (int i = 0; i < bottom_shape[1]; ++i) {
          std::cout << top_data[n*bottom_shape[1]+i] << ", ";
      }
      std::cout << "\n";
  }

  int tmp;
  std:: cin >>  tmp;*/
}

template <typename Dtype>
void LocMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {

    const Dtype* top_diff = top[0]->cpu_diff();            // gradient coming from above
    Dtype* bottom_diff    = bottom[0]->mutable_cpu_diff(); // gradient to be sent below
    //const int count = bottom[0]->count();
    // just copy values from top[0] layer to bottom[0] layer
    caffe_copy(top[0]->count(), top_diff, bottom_diff);

    /*
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
      }*/

  }

}

#ifdef CPU_ONLY
STUB_GPU(LocMaskLayer);
#endif

INSTANTIATE_CLASS(LocMaskLayer);
REGISTER_LAYER_CLASS(LocMask);

}  // namespace caffe
