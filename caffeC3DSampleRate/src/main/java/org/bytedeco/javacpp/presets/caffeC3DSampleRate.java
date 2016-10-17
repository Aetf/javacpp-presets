/*
 * Copyright (C) 2015-2016 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = opencv_highgui.class, target = "org.bytedeco.javacpp.caffeC3DSampleRate",
            value = {
                    @Platform(value = {"linux-x86", "macosx"},
                              define = {"NDEBUG",
                                        "SHARED_PTR_NAMESPACE boost",
                                        "USE_LEVELDB",
                                        "USE_LMDB",
                                        "USE_OPENCV"},
                              include = {
        "google/protobuf/stubs/common.h",
        "google/protobuf/descriptor.h",
        "google/protobuf/message_lite.h",
        "google/protobuf/message.h",
        "caffe/blob.hpp",
        "caffe/caffe.hpp",
        "caffe/common.hpp",
        "caffe/convolution3d_layer.hpp",
        "caffe/data_layers.hpp",
        "caffe/filler.hpp",
        "caffe/layer.hpp",
        "caffe/loss_layers.hpp",
        "caffe/net.hpp",
        "caffe/neuron_layers.hpp",
        "caffe/pool3d_layer.hpp",
        "caffe/proto/caffe.pb.h",
        "caffe/solver.hpp",
        "caffe/syncedmem.hpp",

        "caffe/util/benchmark.hpp",
        "caffe/util/im2col.hpp",
        "caffe/util/image_io.hpp",
        "caffe/util/insert_splits.hpp",
        "caffe/util/io.hpp",
        "caffe/util/math_functions.hpp",
        "caffe/util/mkl_alternate.hpp",
        "caffe/util/rng.hpp",
        "caffe/util/vol2col.hpp",
        "caffe/video_data_layer.hpp",
        "caffe/vision_layers.hpp",
        "caffe/volume_data_layer.hpp"
    },
        link = "caffe",
        includepath = {"/usr/local/cuda/include/", "/System/Library/Frameworks/vecLib.framework/",
        "/System/Library/Frameworks/Accelerate.framework/"},
        linkpath = "/usr/local/cuda/lib/"),
        @Platform(value = {"linux-x86_64", "macosx-x86_64"},
                  define = {"SHARED_PTR_NAMESPACE boost", "USE_LEVELDB", "USE_LMDB", "USE_OPENCV"}) })
public class caffeC3DSampleRate implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LIBPROTOBUF_EXPORT", "LIBPROTOC_EXPORT", "GOOGLE_PROTOBUF_VERIFY_VERSION", "GOOGLE_ATTRIBUTE_ALWAYS_INLINE", "GOOGLE_ATTRIBUTE_DEPRECATED",
                             "GOOGLE_DLOG", "NOT_IMPLEMENTED", "NO_GPU", "CUDA_POST_KERNEL_CHECK").cppTypes().annotations())
               .put(new Info("NDEBUG", "USE_CUDNN", "GFLAGS_GFLAGS_H_", "SWIG").define())
               .put(new Info("defined(_WIN32) && defined(GetMessage)").define(false))
               .put(new Info("cublasHandle_t", "curandGenerator_t").cast().valueTypes("Pointer"))
               .put(new Info("CBLAS_TRANSPOSE", "cublasStatus_t", "curandStatus_t", "hid_t").cast().valueTypes("int"))
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<const google::protobuf::FieldDescriptor*>").pointerTypes("FieldDescriptorVector").define())
               .put(new Info("std::vector<caffe::Datum>").pointerTypes("DatumVector").define())

               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("leveldb::Iterator", "leveldb::DB", "MDB_txn", "MDB_cursor", "MDB_dbi", "MDB_env", "boost::mt19937").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::internal::CompileAssert", "google::protobuf::MessageFactory::InternalRegisterGeneratedFile",
                             "google::protobuf::internal::LogMessage", "google::protobuf::internal::LogFinisher", "google::protobuf::LogHandler",
                             "google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField", "boost::mutex").skip());

        String[] functionTemplates = { "caffe_cpu_gemm", "caffe_cpu_gemv", "caffe_axpy", "caffe_cpu_axpby", "caffe_copy", "caffe_set", "caffe_add_scalar",
                "caffe_scal", "caffe_sqr", "caffe_add", "caffe_sub", "caffe_mul", "caffe_div", "caffe_powx", "caffe_nextafter", "caffe_rng_uniform",
                "caffe_rng_gaussian", "caffe_rng_bernoulli", "caffe_rng_bernoulli", "caffe_exp", "caffe_log", "caffe_abs", "caffe_cpu_dot", "caffe_cpu_strided_dot",
                "caffe_cpu_hamming_distance", "caffe_cpu_asum", "caffe_sign", "caffe_cpu_scale", "caffe_gpu_gemm", "caffe_gpu_gemv", "caffe_gpu_axpy",
                "caffe_gpu_axpby", "caffe_gpu_memcpy", "caffe_gpu_set", "caffe_gpu_memset", "caffe_gpu_add_scalar", "caffe_gpu_scal", "caffe_gpu_add",
                "caffe_gpu_sub", "caffe_gpu_mul", "caffe_gpu_div", "caffe_gpu_abs", "caffe_gpu_exp", "caffe_gpu_log", "caffe_gpu_powx", "caffe_gpu_rng_uniform",
                "caffe_gpu_rng_gaussian", "caffe_gpu_rng_bernoulli", "caffe_gpu_dot", "caffe_gpu_hamming_distance", "caffe_gpu_asum", "caffe_gpu_sign",
                "caffe_gpu_sgnbit", "caffe_gpu_fabs", "caffe_gpu_scale", "hdf5_load_nd_dataset_helper", "hdf5_load_nd_dataset", "hdf5_save_nd_dataset",
                "im2col_nd_cpu", "im2col_cpu", "col2im_nd_cpu", "col2im_cpu", "im2col_nd_gpu", "im2col_gpu", "col2im_nd_gpu", "col2im_gpu" };
        for (String t : functionTemplates) {
            infoMap.put(new Info("caffe::" + t + "<float>").javaNames(t + "_float"))
                   .put(new Info("caffe::" + t + "<double>").javaNames(t + "_double"));
        }

        String classTemplates[] = { "Blob", "Filler", "ConstantFiller", "UniformFiller", "GaussianFiller", "PositiveUnitballFiller", "XavierFiller", "MSRAFiller", "BilinearFiller",
                "BaseDataLayer", "Batch", "BasePrefetchingDataLayer", "DataLayer", "DummyDataLayer", "HDF5DataLayer", "HDF5OutputLayer", "ImageDataLayer", "MemoryDataLayer",
                "WindowDataLayer", "Layer", "AccuracyLayer", "LossLayer", "ContrastiveLossLayer", "EuclideanLossLayer", "HingeLossLayer",
                "InfogainLossLayer", "MultinomialLogisticLossLayer", "SigmoidCrossEntropyLossLayer", "SoftmaxWithLossLayer", "NeuronLayer", "AbsValLayer", "BNLLLayer",
                "DropoutLayer", "ExpLayer", "PowerLayer", "ReLULayer", "SigmoidLayer", "TanHLayer", "ThresholdLayer", "PReLULayer", "PythonLayer", "ArgMaxLayer", "BatchNormLayer",
                "BatchReindexLayer", "ConcatLayer", "EltwiseLayer", "EmbedLayer", "FilterLayer", "FlattenLayer", "InnerProductLayer", "MVNLayer", "ReshapeLayer", "ReductionLayer",
                "SilenceLayer", "SoftmaxLayer", "SplitLayer", "SliceLayer", "TileLayer", "Net", "Solver", "WorkerSolver", "SolverRegistry", "SolverRegisterer", "SGDSolver", "NesterovSolver",
                "AdaGradSolver", "RMSPropSolver", "AdaDeltaSolver", "AdamSolver",  "InputLayer", "ParameterLayer", "BaseConvolutionLayer", "ConvolutionLayer", "CropLayer", "DeconvolutionLayer",
                "Im2colLayer", "LRNLayer", "PoolingLayer", "SPPLayer", "RecurrentLayer", "LSTMLayer", "RNNLayer"
                /* "CuDNNReLULayer", "CuDNNSigmoidLayer", "CuDNNTanHLayer", "CuDNNSoftmaxLayer", "CuDNNConvolutionLayer", "CuDNNPoolingLayer" */ };
        for (String t : classTemplates) {
            boolean purify = t.equals("BaseDataLayer") || t.equals("LossLayer") || t.equals("NeuronLayer");
            boolean virtualize = t.endsWith("Layer") || t.endsWith("Solver");
            infoMap.put(new Info("caffe::" + t + "<float>").pointerTypes("Float" + t).purify(purify).virtualize(virtualize))
                   .put(new Info("caffe::" + t + "<double>").pointerTypes("Double" + t).purify(purify).virtualize(virtualize));
        }
        infoMap.put(new Info("caffe::BasePrefetchingDataLayer<float>::InternalThreadEntry()",
                             "caffe::BasePrefetchingDataLayer<double>::InternalThreadEntry()").skip())

               .put(new Info("caffe::Solver<float>::Solve(std::string)", "caffe::Solver<double>::Solve(std::string)").javaText(
                             "public void Solve(String resume_file) { Solve(new BytePointer(resume_file)); }\n"
                           + "public void Solve() { Solve((BytePointer)null); }"))

               .put(new Info("caffe::Batch<float>::data_").javaText("@MemberGetter public native @ByRef FloatBlob data_();"))
               .put(new Info("caffe::Batch<double>::data_").javaText("@MemberGetter public native @ByRef DoubleBlob data_();"))
               .put(new Info("caffe::Batch<float>::label_").javaText("@MemberGetter public native @ByRef FloatBlob label_();"))
               .put(new Info("caffe::Batch<double>::label_").javaText("@MemberGetter public native @ByRef DoubleBlob label_();"))

               .put(new Info("caffe::GetFiller<float>").javaNames("GetFloatFiller"))
               .put(new Info("caffe::GetFiller<double>").javaNames("GetDoubleFiller"))
               .put(new Info("caffe::GetSolver<float>").javaNames("GetFloatSolver"))
               .put(new Info("caffe::GetSolver<double>").javaNames("GetDoubleSolver"))

               .put(new Info("boost::shared_ptr<caffe::Blob<float> >").annotations("@SharedPtr").pointerTypes("FloatBlob"))
               .put(new Info("boost::shared_ptr<caffe::Blob<double> >").annotations("@SharedPtr").pointerTypes("DoubleBlob"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<float> > >").pointerTypes("FloatBlobSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Blob<double> > >").pointerTypes("DoubleBlobSharedVector").define())

               .put(new Info("boost::shared_ptr<caffe::Layer<float> >").annotations("@Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr").pointerTypes("FloatLayer"))
               .put(new Info("boost::shared_ptr<caffe::Layer<double> >").annotations("@Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr").pointerTypes("DoubleLayer"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Layer<float> > >").pointerTypes("FloatLayerSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Layer<double> > >").pointerTypes("DoubleLayerSharedVector").define())

               .put(new Info("boost::shared_ptr<caffe::Net<float> >").annotations("@SharedPtr").pointerTypes("FloatNet"))
               .put(new Info("boost::shared_ptr<caffe::Net<double> >").annotations("@SharedPtr").pointerTypes("DoubleNet"))
               .put(new Info("std::vector<boost::shared_ptr<caffe::Net<float> > >").pointerTypes("FloatNetSharedVector").define())
               .put(new Info("std::vector<boost::shared_ptr<caffe::Net<double> > >").pointerTypes("DoubleNetSharedVector").define())

               .put(new Info("std::vector<caffe::Blob<float>*>").pointerTypes("FloatBlobVector").define())
               .put(new Info("std::vector<caffe::Blob<double>*>").pointerTypes("DoubleBlobVector").define())
               .put(new Info("std::vector<std::vector<caffe::Blob<float>*> >").pointerTypes("FloatBlobVectorVector").define())
               .put(new Info("std::vector<std::vector<caffe::Blob<double>*> >").pointerTypes("DoubleBlobVectorVector").define())

               .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
               .put(new Info("std::vector<std::vector<bool> >").pointerTypes("BoolVectorVector").define())
               .put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define())

               .put(new Info("caffe::Net<float>::layer_by_name").javaText(
                       "public FloatLayer layer_by_name(BytePointer layer_name) { return layer_by_name(FloatLayer.class, layer_name); }\n"
                     + "public FloatLayer layer_by_name(String layer_name) { return layer_by_name(FloatLayer.class, layer_name); };\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<float> >\"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);\n"))
               .put(new Info("caffe::Net<double>::layer_by_name").javaText(
                       "public DoubleLayer layer_by_name(BytePointer layer_name) { return layer_by_name(DoubleLayer.class, layer_name); }\n"
                     + "public DoubleLayer layer_by_name(String layer_name) { return layer_by_name(DoubleLayer.class, layer_name); };\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);\n"
                     + "public native @Const @Cast({\"\", \"boost::shared_ptr<caffe::Layer<double> >\"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);\n"));
    }
}
