#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace boost::filesystem;

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  string usage_message = "Compute the mean_image of a set of images in a folder\nUsage:\ncompute_image_mean_folder <input_folder> <output_file_name.binaryproto>\n";

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3) {
    std::cout << usage_message;
    return 1;
  }

  if(!exists(argv[1]) || !is_directory(argv[1])) return -1;
  recursive_directory_iterator it(argv[1]);
  recursive_directory_iterator endit;
  vector<string> file_paths;

  while(it != endit) {
    if(is_regular_file(*it) && (it->path().extension() == ".png" || it->path().extension() == ".jpg")) file_paths.push_back(it->path().string());
    ++it;
  }

  /*
  for(int index = 0; index < file_paths.size(); index++)
    std::cout << file_paths[index] << "\n";
  */

  std::cout << "Found a total of " << file_paths.size() << " files\n";  


  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Datum datum;
  ReadImageToDatum(file_paths[0], 1, &datum);
  DecodeDatumNative(&datum);
  
  std::cout << "Channels: " << datum.channels() << ", Height: " << datum.height() << ", Width: " << datum.width() << "\n";

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  for(int index = 0; index < file_paths.size(); index++) {
    Datum datum;
    ReadImageToDatum(file_paths[index], 1, &datum);
    DecodeDatumNative(&datum);

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
  return 0;
}
