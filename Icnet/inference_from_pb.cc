#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <dirent.h>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace tensorflow;
//using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using namespace cv;

int TensorFromFile(string filename, const int i_height, const int i_width, std::vector<Tensor> *o_tensors)
{
  tensorflow::Status status;
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));
  tensorflow::GraphDef graph;

  auto reader = tensorflow::ops::ReadFile(root.WithOpName("img_reader"), filename);
  const int channels = 3;
  tensorflow::Output imgreader;

  if (tensorflow::str_util::EndsWith(filename, ".png"))
  {
    imgreader = DecodePng(root.WithOpName("png_reader"), reader, DecodePng::Channels(channels));
  }
  else if (tensorflow::str_util::EndsWith(filename, ".gif"))
  {
    imgreader = DecodeGif(root.WithOpName("gif_reader"), reader);
  }
  else
  {
    imgreader = DecodeJpeg(root.WithOpName("jpeg_reader"), reader, DecodeJpeg::Channels(channels));
  }

  auto uint8_caster = Cast(root.WithOpName("output"), imgreader, tensorflow::DT_UINT8);
  ExpandDims(root.WithOpName("output"), uint8_caster, 0);

  status = root.ToGraphDef(&graph);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  status = session->Create(graph);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  status = session->Run({}, {"output"}, {}, o_tensors);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  return 0;
}

int TensorToFile(string filename, std::vector<Tensor> &out, float threshold = 0.5f)
{
  using namespace ::tensorflow::ops;
  auto root = tensorflow::Scope::NewRootScope();
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));
  tensorflow::GraphDef graph;
  tensorflow::Status status;

  auto quantlevel = Const(root.WithOpName("quantization"), 255.0f);
  auto f_caster = Cast(root.WithOpName("float_caster"), out[0], tensorflow::DT_FLOAT);
  auto mul = Mul(root.WithOpName("multiple_quantization"), f_caster, quantlevel);
  auto cast = Cast(root.WithOpName("cast"), mul, DT_UINT8);
  auto squeeze = Squeeze(root.WithOpName("squeezeDim"), cast, Squeeze::Attrs().Axis({0}));
  auto encode = EncodePng(root.WithOpName("pngencoder"), squeeze);
  auto writefile = tensorflow::ops::WriteFile(root.WithOpName("writer"), filename, encode);

  status = root.ToGraphDef(&graph);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  status = session->Create(graph);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  status = session->Run({}, {}, {"writer"}, nullptr);
  if (!status.ok())
  {
    LOG(ERROR) << status.ToString();
    return -1;
  }

  session->Close();
  return 0;
}

int main(int argc, char *argv[])
{
  using namespace ::tensorflow::ops;
  tensorflow::Status status;
  std::string delimiter = ".";
  std::string output_filename;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::string mdlpath;
  std::cout << "Enter PB file path: " << std::endl;
  std::cin >> mdlpath;
  std::string input_folder;
  std::cout << "Enter input_folder path: " << std::endl;
  std::cin >> input_folder;
  std::string output_folder;
  std::cout << "Enter output_folder path: " << std::endl;
  std::cin >> output_folder;
  int32 inputheight;
  int32 inputwidth;
  float threshold = atof("0.5");

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));

  tensorflow::GraphDef graph;

  status = ReadBinaryProto(Env::Default(), mdlpath, &graph);
  if (!status.ok())
  {
    std::cout << status.ToString() << "\n";
    return -1;
  }

  status = session->Create(graph);
  if (!status.ok())
  {
    std::cout << status.ToString() << "\n";
    return -1;
  }

  std::vector<std::string> ImageFiles;
  DIR *ImageDIR;
  struct dirent *entry;
  if (ImageDIR = opendir(input_folder.c_str()))
  {
    while (entry = readdir(ImageDIR))
    {
      if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        ImageFiles.push_back(entry->d_name);
    }
  }
  closedir(ImageDIR);
  std::sort(ImageFiles.begin(), ImageFiles.end());

  for (int i = 0; i < ImageFiles.size(); i++)
  {
    std::string imgpath = input_folder + "/" + ImageFiles[i];

    if (TensorFromFile(imgpath, inputheight, inputwidth, &inputs))
    {
      LOG(ERROR) << "Image reading failed"
                 << "\n";
      return -1;
    }

    auto inputlayer = "inputs";
    auto outputlayer = "predictions";

    status = session->Run({{inputlayer, inputs[0]}}, {outputlayer}, {}, &outputs);

    std::cout << status << std::endl;
    if (!status.ok())
    {
      LOG(ERROR) << status.ToString();
      return -1;
    }

    output_filename = output_folder + "/" + ImageFiles[i];

    std::cout << "output filename: " << output_filename << "\n";

    //Now write this to a image file
    if (TensorToFile(output_filename, outputs, threshold))
      return -1;
  }
  session->Close();
  return 0;
}
