// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna2_model_helper.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#if GNA_LIB_VER == 2
#include "gna2_model_debug_log.hpp"
#include "gna2-model-api.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <sstream>
#include <string>
#include <cmath>
#include <map>

std::string GetLayerType(Gna2OperationType type) {
    switch (type) {
    case Gna2OperationTypeFullyConnectedAffine: return "Gna2OperationTypeFullyConnectedAffine";
    case Gna2OperationTypeElementWiseAffine: return "Gna2OperationTypeElementWiseAffine";
    case Gna2OperationTypeRecurrent: return "Gna2OperationTypeRecurrent";
    case Gna2OperationTypeConvolution: return "Gna2OperationTypeConvolution";
    case Gna2OperationTypeTransposition: return "Gna2OperationTypeTransposition";
    case Gna2OperationTypeCopy: return "Gna2OperationTypeCopy";
    default: return "Gna2OperationUNKNOWN";
    }
}

void write_pwl(std::ostream & pwl_file, const Gna2PwlSegment* const segments, const uint32_t numberOfSegments) {
    for (uint32_t k = 0; k < numberOfSegments; k++) {
        pwl_file << segments[k].Slope << ", " << segments[k].xBase << ", " << segments[k].yBase << "\n";
    }
}

void write_pwl(std::ostream & pwl_file, const Gna2Tensor& activation) {
    write_pwl(pwl_file, static_cast<Gna2PwlSegment*>(activation.Data), activation.Shape.Dimensions[0]);
}

std::string GetSimpleString(Gna2Shape shape) {
    std::stringstream out;
    for (uint32_t i = 0; i < shape.NumberOfDimensions; i++) {
        out << shape.Dimensions[i];
        if (i + 1 < shape.NumberOfDimensions) out << 'x';
    }
    return out.str();
}

template <class T>
bool NextElement(T & elementIndex, const Gna2Shape& total) {
    if (total.NumberOfDimensions == 0) return false;
    auto idx = total.NumberOfDimensions - 1;
    IE_ASSERT(idx < GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS);
    while (elementIndex[idx] + 1 >= total.Dimensions[idx] && idx > 0) {
        idx--;
    }
    if (elementIndex[idx] + 1 >= total.Dimensions[idx] && idx == 0) {
        return false;
    }
    ++(elementIndex[idx]);
    while (++idx < total.NumberOfDimensions) {
        elementIndex[idx] = 0;
    }
    return true;
}

template <class T>
uint32_t GetLinearIndex(const T & elementIndex, const Gna2Shape& total) {
    uint32_t out = 0;
    for (int idx = 0; idx < total.NumberOfDimensions; idx++) {
        out += elementIndex[idx];
        if (idx + 1 < total.NumberOfDimensions) out *= total.Dimensions[idx + 1];
    }
    return out;
}

template <class T>
int32_t GetValue(const Gna2Tensor& tensor, const T & elementIndex) {
    int32_t intValue = -123;  // fake value indicating problem
    const auto linearIndex = GetLinearIndex(elementIndex, tensor.Shape);
    if (tensor.Type == Gna2DataTypeInt32) {
        intValue = (reinterpret_cast<int32_t *>(tensor.Data)[linearIndex]);
    } else if (tensor.Type == Gna2DataTypeInt16) {
        intValue = reinterpret_cast<int16_t *>(tensor.Data)[linearIndex];
    } else {
        intValue = reinterpret_cast<int8_t*>(tensor.Data)[linearIndex];
    }
    return intValue;
}

void WriteInputAndOutputTextGNAImpl(const Gna2Model & gnaModel, const std::string dumpFolderNameGNA, const std::string refFolderName) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        const auto & operation = gnaModel.Operations[i];
        std::stringstream out_file_name;
        const auto & outputTensor = *operation.Operands[OutOpIdx];
        const auto & intputTensor = *operation.Operands[InOpIdx];

        out_file_name << std::setfill('0') << std::setw(2) << i << "_"
            << GetLayerType(operation.Type)
            << "-" << GetSimpleString(intputTensor.Shape)
            << "-" << GetSimpleString(outputTensor.Shape);

        auto inputfileName = dumpFolderNameGNA + out_file_name.str() + "_input.txt";
        auto outFileName = dumpFolderNameGNA + out_file_name.str() + "_output.txt";
        auto pwlFileName = dumpFolderNameGNA + out_file_name.str() + "_pwl.txt";
        auto refOutputFileName = refFolderName + out_file_name.str() + "_output.txt";

        std::ofstream out_file(outFileName.c_str(), std::ios::out);
        std::ofstream pwl_file(pwlFileName.c_str(), std::ios::out);
        std::ifstream ref_out_file(refOutputFileName.c_str(), std::ios::in);
        std::ofstream in_file(inputfileName.c_str(), std::ios::out);

        float  summOfDiff = 0.f;
        float  summOfSqDiff = 0.f;
        float  maxD = 0.0f;
        int    numItems = 0;

        if (operation.NumberOfOperands > PwlOpIdx && operation.Operands[PwlOpIdx] != nullptr) {
            write_pwl(pwl_file, *operation.Operands[PwlOpIdx]);
        }

        std::vector<uint32_t> elementIndex(outputTensor.Shape.NumberOfDimensions);

        do {
            float floatValue = GetValue(outputTensor, elementIndex);

            out_file << std::setw(8) << floatValue << "\n";
            if (ref_out_file) {
                float ref_value = 0.f;
                ref_out_file >> ref_value;
                float diff = (ref_value - floatValue);
                diff = diff < 0 ? -diff : diff;
                summOfDiff += diff;
                summOfSqDiff += diff * diff;
                maxD = std::max(maxD, diff);
                numItems++;
            }
        } while (NextElement(elementIndex, outputTensor.Shape));


        if (numItems) {
            auto rmse = std::sqrt(summOfSqDiff / numItems);
            auto avg = summOfDiff / numItems;
            std::cout << std::left << std::setw(55) << out_file_name.str()
                << " RMSE=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << rmse
                << " avg=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << avg
                << " maxD=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << maxD << std::endl;
        }

        std::vector<uint32_t> inputElementIndex(intputTensor.Shape.NumberOfDimensions);

        do {
            int32_t intValue = GetValue(intputTensor, inputElementIndex);
            in_file << std::setw(8) << intValue;
            in_file << "\n";
        } while (NextElement(inputElementIndex, intputTensor.Shape));
    }
}

static std::string GetOperandType(Gna2DataType type) {
    const std::map<Gna2DataType, std::string> operandTypeMap = {
        {Gna2DataTypeNone, "Gna2DataTypeNone"},
        {Gna2DataTypeBoolean, "Gna2DataTypeBoolean"},
        {Gna2DataTypeInt4, "Gna2DataTypeInt4"},
        {Gna2DataTypeInt8, "Gna2DataTypeInt8"},
        {Gna2DataTypeInt16, "Gna2DataTypeInt16"},
        {Gna2DataTypeInt32, "Gna2DataTypeInt32"},
        {Gna2DataTypeUint4, "Gna2DataTypeUint4"},
        {Gna2DataTypeUint8, "Gna2DataTypeUint8"},
        {Gna2DataTypeUint16, "Gna2DataTypeUint16"},
        {Gna2DataTypeUint32, "Gna2DataTypeUint32"},
        {Gna2DataTypeUint64, "Gna2DataTypeUint64"},
        {Gna2DataTypeCompoundBias, "Gna2DataTypeCompoundBias"},
        {Gna2DataTypePwlSegment, "Gna2DataTypePwlSegment"},
        {Gna2DataTypeWeightScaleFactor, "Gna2DataTypeWeightScaleFactor"}
    };
    return operandTypeMap.find(type)->second;
}

static std::vector<std::string> GetOperandNames(Gna2OperationType type) {
    const std::map<Gna2OperationType, std::vector<std::string>> operationOperandNamesMap = {
        {Gna2OperationTypeConvolution, {"inputs", "outputs", "filters", "biases", "activationFunction"}},
        {Gna2OperationTypeCopy, {"inputs", "outputs"}},
        {Gna2OperationTypeFullyConnectedAffine, {"inputs", "outputs", "weights", "biases", "activationFunction", "weightScaleFactors"}},
        {Gna2OperationTypeElementWiseAffine, {"inputs", "outputs", "weights", "biases", "activationFunction"}},
        {Gna2OperationTypeGmm, {"inputs", "outputs", "means/interleaved", "inverseCovariances", "constants"}},
        {Gna2OperationTypeRecurrent, {"inputs", "outputs", "weights", "biases", "activationFunction"}},
        {Gna2OperationTypeTransposition, {"inputs", "outputs"}}
    };
    return operationOperandNamesMap.find(type)->second;
}

static std::vector<std::string> GetParamaterNames(Gna2OperationType type) {
    const std::map<Gna2OperationType, std::vector<std::string>> operationParamaterNamesMap = {
        {Gna2OperationTypeConvolution, {"convolutionStride", "biasMode", "poolingMode", "poolingWindow", "poolingStride", "zeroPadding"}},
        {Gna2OperationTypeCopy, {"shape (sub-tensor shape)"}},
        {Gna2OperationTypeFullyConnectedAffine, {"biasMode", "biasVectorIndex"}},
        {Gna2OperationTypeGmm, {"maximumScore"}},
        {Gna2OperationTypeRecurrent, {"delay"}}
    };
    return operationParamaterNamesMap.find(type)->second;
}

static std::string GetBiasMode(Gna2BiasMode mode) {
    const std::map<Gna2BiasMode, std::string> biasModeMap = {
        {Gna2BiasModeDefault, "Gna2BiasModeDefault"},
        {Gna2BiasModePerStride, "Gna2BiasModePerStride"},
        {Gna2BiasModeGrouping, "Gna2BiasModeGrouping"}
    };
    return biasModeMap.find(mode)->second;
}

static std::string GetPoolingMode(Gna2PoolingMode mode) {
    const std::map<Gna2PoolingMode, std::string> poolingModeMap = {
        {Gna2PoolingModeDisabled, "Gna2PoolingModeDisabled"},
        {Gna2PoolingModeMax, "Gna2PoolingModeMax"},
        {Gna2PoolingModeSum, "Gna2PoolingModeSum"}
    };
    return poolingModeMap.find(mode)->second;
}

static void DumpShape(std::ostream& dumpFile, Gna2Shape* shape, const std::string paramName) {
    dumpFile << "\tParameter name: " << paramName << ", ";
    dumpFile << "parameter type: Gna2Shape\n";
    dumpFile << "\t\tNumber of dimensions: " << shape->NumberOfDimensions;
    dumpFile << "\n\t\tDimensions: [";
    for (uint32_t i = 0; i < shape->NumberOfDimensions; i++) {
        dumpFile << std::setw(8) << shape->Dimensions[i];
    }
    dumpFile << "]\n";
}

static void DumpConvolutionParameters(std::ostream& dumpFile, void** parameters, size_t knownParamCount, const std::vector<std::string> paramNames) {
    size_t i = 0;

    while (i < knownParamCount) {
        if (i == ConvStrideParamIdx || i == PoolWinParamIdx || i == PoolStrideParamIdx || i == ZeroPaddingParamIdx) {
            Gna2Shape* shape = reinterpret_cast<Gna2Shape*>(parameters[i]);
            if (shape != nullptr)
                DumpShape(dumpFile, shape, paramNames[i]);
        } else if (i == BiasModeCnnParamIdx) {
            Gna2BiasMode* biasMode = reinterpret_cast<Gna2BiasMode*>(parameters[i]);
            if (biasMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetBiasMode(*biasMode) << "\n";
        } else {
            Gna2PoolingMode* poolingMode = reinterpret_cast<Gna2PoolingMode*>(parameters[i]);
            if (poolingMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetPoolingMode(*poolingMode) << "\n";
        }
        i++;
    }
 }

static void DumpCopyParameters(std::ostream& dumpFile, void** parameters, size_t knownParamCount, const std::vector<std::string> paramNames) {
        Gna2Shape* subTensorShape = reinterpret_cast<Gna2Shape*>(parameters[CopyShapeParamIdx]);
        DumpShape(dumpFile, subTensorShape, paramNames[CopyShapeParamIdx]);
}

static void DumpFCAffineParameters(std::ostream& dumpFile, void** parameters, size_t knownParamCount, const std::vector<std::string> paramNames) {
    size_t i = 0;

    while (i < knownParamCount) {
        if (i == BiasModeFCAffineParamIdx) {
            Gna2BiasMode* biasMode = reinterpret_cast<Gna2BiasMode*>(parameters[BiasModeFCAffineParamIdx]);
            if (biasMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetBiasMode(*biasMode) << "\n";
        } else {
            uint32_t* biasVectorIndex = reinterpret_cast<uint32_t*>(parameters[BiasModeFCAffineParamIdx]);
            if (biasVectorIndex != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << *biasVectorIndex << "\n";
        }
        i++;
    }
}

static void DumpIntParameter(std::ostream& dumpFile, void** parameters, size_t knownParamCount, const std::vector<std::string> paramNames) {
    uint32_t* param = reinterpret_cast<uint32_t*>(parameters[0]);
    if (param != nullptr)
        dumpFile << "\tParameter name: " << paramNames[0] << ", value: " << *param << "\n";
}

typedef void (*dumpParamaters) (std::ostream&, void**, size_t, const std::vector<std::string>);

static dumpParamaters GetParamDumpFunc(Gna2OperationType type) {
    static const std::map<Gna2OperationType, dumpParamaters> dumpParamMap = {
        {Gna2OperationTypeConvolution, DumpConvolutionParameters},
        {Gna2OperationTypeCopy, DumpCopyParameters},
        {Gna2OperationTypeFullyConnectedAffine, DumpFCAffineParameters},
        {Gna2OperationTypeGmm, DumpIntParameter},
        {Gna2OperationTypeRecurrent, DumpIntParameter}
    };
    return dumpParamMap.find(type) != dumpParamMap.end() ? dumpParamMap.find(type)->second : nullptr;
}

static void DumpPwl(std::ostream& dumpFile, const Gna2Tensor& activation) {
    const Gna2PwlSegment* const segments = static_cast<Gna2PwlSegment*>(activation.Data);
    const uint32_t numberOfSegments = activation.Shape.Dimensions[0];

    for (uint32_t k = 0; k < numberOfSegments; k++) {
        uint32_t scale = ((segments[k].xBase & 3) + 1) * 8;
        uint64_t factor = 1ULL << scale;
        int32_t B = segments[k].xBase & 0xfffffffc;
        double a = static_cast<double>(segments[k].Slope) / factor;
        double b = static_cast<double>(segments[k].yBase) - ((static_cast<double>(B) * segments[k].Slope) / factor);

        dumpFile << "\t\tBase value for input (B) : " << B << "\n";
        dumpFile << "\t\tBase value for output (b) : " << segments[k].yBase << "\n";
        dumpFile << "\t\tSegment slope (S): " << segments[k].Slope << "\n";
        dumpFile << "\t\tShift (scale) : " << scale << "\n";
        dumpFile << "\t\ty = ax + b:   a = " << a << ", b = " << b;
        if (segments[k].Slope != 0) {
            double x0 = static_cast<double>(B) - ((static_cast<double>(segments[k].yBase) * factor) / segments[k].Slope);
            dumpFile << ", x0 = " << x0;
        }
        dumpFile << "\n";
    }
}

static void DumpCompoundBias(std::ostream& dumpFile, const Gna2Tensor& tensor) {
    auto i = 0;

    while (i < tensor.Shape.Dimensions[0]) {
        const Gna2CompoundBias* const bias = static_cast<Gna2CompoundBias*>(tensor.Data) + i;
        dumpFile << "\t\tBias for row " << i << " : " << bias->Bias << ", multiplier: " << unsigned(bias->Multiplier) << "\n";
        i++;
    }
}

static void DumpCharArray(std::ostream& dumpFile, const char *carray,  size_t count) {
    auto i = 0;
    while (*(carray + i) != 0 && i < count) {
        dumpFile << *(carray + i) << " ";
        i++;
    }
    dumpFile << "\n";
}

void DumpGna2Model(const Gna2Model& gnaModel, const std::string dumpFolderNameGNA, bool dumpData) {
    std::stringstream dumpFileName;
    uint32_t opsNo = gnaModel.NumberOfOperations;
    std::time_t currTime = std::time(nullptr);

    dumpFileName << dumpFolderNameGNA << "Gna2ModelDebugDump_" << opsNo << "_layer_" << std::put_time(std::localtime(&currTime), "%Y%m%d%H%M%S");

    std::ofstream dumpFile(dumpFileName.str() + ".txt", std::ios::out);

    dumpFile << "Layers (operations) count: " << opsNo << "\n";

    for (uint32_t i = 0; i < opsNo; i++) {
        const auto& operation = gnaModel.Operations[i];

        dumpFile << "------------------------------------------------------------------------\n\n";

        dumpFile << "Layer (operation): " << i << "\n";
        dumpFile << "Layer (operation) type: " << GetLayerType(operation.Type) << "\n";
        dumpFile << "Number of possible operands: " << operation.NumberOfOperands << "\n";

        for (uint32_t j = 0; j < operation.NumberOfOperands; j++) {
            if (operation.Operands[j] == nullptr) {
                dumpFile << "\tOperand " << j << " == nullptr\n";
                continue;
            }
            const auto& operand = *operation.Operands[j];
            dumpFile << "\tOperand " << j << " (" << GetOperandNames(operation.Type)[j] << ")"
                << " type: " << GetOperandType(operand.Type) <<
                " shape: " << GetSimpleString(operand.Shape) <<
                " layout: ";

            DumpCharArray(dumpFile, operand.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS);

            if (operand.Type == Gna2DataTypePwlSegment) {
                DumpPwl(dumpFile, operand);
            } else if (operand.Type == Gna2DataTypeCompoundBias) {
                DumpCompoundBias(dumpFile, operand);
            } else if (dumpData) {
                std::ofstream datFile(dumpFileName.str() + ".dat", std::ios::out);
                std::vector<uint32_t> elementIndex(operand.Shape.NumberOfDimensions);

                datFile << "Layer " << i << ", type " << GetLayerType(operation.Type) <<
                    ", operand " << j << " - " << GetOperandNames(operation.Type)[j] << "\n";

                do {
                    int32_t value = GetValue(operand, elementIndex);
                    datFile << value << "\n";
                } while (NextElement(elementIndex, operand.Shape));
            }
        }

        dumpFile << "Parameters: \n";

        if (operation.NumberOfParameters > 0 && GetParamDumpFunc(operation.Type) != nullptr) {
            std::vector<std::string> paramNames = GetParamaterNames(operation.Type);
            size_t knownParamCount = operation.NumberOfParameters <= paramNames.size() ? operation.NumberOfParameters : paramNames.size();
            GetParamDumpFunc(operation.Type)(dumpFile, operation.Parameters, knownParamCount, paramNames);
        }
    }
}
#endif
