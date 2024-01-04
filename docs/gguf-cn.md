# GGUF

GGUF是一种用于存储GGML模型和基于GGML执行器的文件格式。GGUF是一种二进制格式，旨在实现模型的快速加载和保存，以及便于阅读。传统上，模型是使用PyTorch或其他框架开发的，然后转换为GGUF以在GGML中使用。

它是GGML、GGMF和GGJT的后继文件格式，旨在通过包含加载模型所需的所有信息而变得明确。它还被设计为可扩展的，可以向模型添加新信息而不会破坏与现有模型的兼容性。

有关GGUF背后动机的更多信息，请参阅[历史状态](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#historical-state-of-affairs)。

## 规范

GGUF是基于现有GGJT的格式，但对格式进行了一些更改，使其更具扩展性和易用性。以下功能是期望的：

- 单文件部署：它们可以轻松分发和加载，并且不需要任何外部文件来提供附加信息。
- 可扩展性：可以向基于GGML的执行器/向GGUF模型添加新特性/新信息，而不会破坏与现有模型的兼容性。
- `mmap` 兼容性：可以使用`mmap`快速加载和保存模型。
- 易于使用：可以使用少量代码轻松加载和保存模型，无论使用的语言是什么，无需外部库。
- 全部信息：加载模型所需的所有信息都包含在模型文件中，用户无需提供任何其他信息。

GGJT和GGUF之间的关键区别在于超参数（现在称为元数据）采用了键值结构，而不是未经类型化值的列表。这允许向现有模型添加新元数据而不会破坏与现有模型的兼容性，并且可以使用附加信息对模型进行注释，这些信息可能对推断或识别模型有用。

### 文件结构

GGUF文件结构如下所示。它们使用在 `general.alignment` 元数据字段中指定的全局对齐，以下简称为 `ALIGNMENT`。在必要时，文件使用 `0x00` 字节填充到下一个 `general.alignment` 的倍数。

字段，包括数组，按顺序写入，除非另有规定，否则不进行对齐。

模型默认为小端字节序。它们也可以以大端字节序提供，用于与大端字节序计算机一起使用；在这种情况下，所有值（包括元数据值和张量）也将是大端字节序的。在撰写时，无法确定模型是否为大端字节序；这可能在未来版本中得到纠正。如果没有提供其他信息，请假定模型为小端字节序。

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
}

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
}

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

## 标准的kv对

以下是标准化的键值对列表。随着发现更多用例，这个列表可能会不断增长。在可能的情况下，名称与原始模型定义共享，以便更轻松地在两者之间进行映射。

并非所有这些都是必需的，但都是推荐的。加粗的键是必需的。对于省略的键值对，读者应假设值是未知的，并根据需要默认或报错处理。

社区可以开发自己的键值对以携带额外数据。然而，这些应该使用相关社区名称进行命名空间分隔，以避免冲突。例如，`rustformers` 社区可能使用 `rustformers.` 作为所有键的前缀。

如果特定的社区键被广泛使用，它可能会被提升为标准化键。

按照惯例，大多数计数/长度等都是 `uint64`，除非另有规定。这是为了将来支持更大的模型。一些模型可能使用 `uint32` 作为它们的值；建议读者同时支持这两种。

### 通用

#### 必需

- **`general.architecture: string`**

  : 描述此模型实现的架构。全部为小写ASCII字符，仅允许

   

  ```
  [a-z0-9]+
  ```

   

  。已知的值包括：

  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `rwkv`

- **`general.quantization_version: uint32`**：量化格式的版本。如果模型未量化（即没有张量被量化），则不需要。如果有任何张量被量化，*必须*存在。这是与张量本身的量化方案分开的；量化版本可以更改而不更改方案的名称（例如，量化方案是Q5_K，量化版本是4）。

- **`general.alignment: uint32`**：使用的全局对齐，如上所述。这可以变化以允许不同的对齐方案，但必须是8的倍数。某些编写器可能不会写入对齐。如果**没有**指定对齐，假设它为 `32`。

#### 通用元数据

- `general.name`：模型的名称。这应该是一个可供人们识别模型的人类可读名称。它应该在定义模型的社区内是唯一的。

- `general.author`：模型的作者。

- `general.url`：模型主页的URL。这可以是GitHub存储库、论文等。

- `general.description: string`：模型的自由形式描述，包括其他字段未涵盖的任何内容。

- `general.license: string`：模型的许可证，以[SPDX许可证表达式](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/)表示（例如，`"MIT OR Apache-2.0`）。不包括任何其他信息，如许可证文本或许可证的URL。

- ```
  general.file_type: uint32
  ```

  ：描述文件中大多数张量类型的枚举值。可选；可以从张量类型中推断出。 

  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` (支持已移除)
  - `MOSTLY_Q4_3 = 6` (支持已移除)
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### 源元数据

有关模型来源的信息。这对于跟踪模型的来源以及在模型被修改时找到原始来源是有用的。例如，对于从GGML转换的模型，这些键将指向从中转换而来的模型。

- `general.source.url: string`：模型来源的URL。可以是GitHub存储库、论文等。
- `general.source.huggingface.repository: string`：Hugging Face模型存储库，此模型是基于其上托管的或基于其上构建的。

### LLM

以下是LLM架构中可用的键值对。其中 `[llm]` 用于表示特定LLM架构的名称。例如，`llama` 代表LLaMA，`mpt` 代表MPT等。如果在架构的部分提到，那么对于该架构来说，这些键是必需的，但并非所有键都对所有架构都是必需的。请查阅相关部分以获取更多信息。

- `[llm].context_length: uint64`：也称为 `n_ctx`。模型训练时上下文的长度（以标记表示）。对于大多数架构，这是输入长度的硬限制。并非依赖于变压器式注意力的架构（例如RWKV）可能能够处理更大的输入，但这并不保证。
- `[llm].embedding_length: uint64`：也称为 `n_embd`。嵌入层大小。
- `[llm].block_count: uint64`：注意力+前馈层块的数量（即LLM的主体部分）。不包括输入或嵌入层。
- `[llm].feed_forward_length: uint64`：也称为 `n_ff`。前馈层的长度。
- `[llm].use_parallel_residual: bool`：是否应使用并行残差逻辑。
- `[llm].tensor_data_layout: string`：当将模型转换为GGUF时，张量可以重新排列以提高性能。此键描述了张量数据的布局。这不是必需的；如果不存在，则假定为 `reference`。
  - `reference`：张量按照原始模型的顺序排列
  - 可以在各自架构的部分找到更多选项
- `[llm].expert_count: uint32`：MoE模型中的专家数量（对于非MoE架构是可选的）。
- `[llm].expert_used_count: uint32`：每个标记评估时使用的专家数量（对于非MoE架构是可选的）。

#### 注意力

- `[llm].attention.head_count: uint64`：也称为 `n_head`。注意力头的数量。
- `[llm].attention.head_count_kv: uint64`：在分组查询注意力中使用的每组的头数。如果不存在，或者存在且等于 `[llm].attention.head_count`，则模型不使用GQA。
- `[llm].attention.max_alibi_bias: float32`：用于ALiBI的最大偏置。
- `[llm].attention.clamp_kqv: float32`：用于将 `Q`、`K` 和 `V` 张量的值夹在 (`[-C, C]`) 区间的值（`C`）。
- `[llm].attention.layer_norm_epsilon: float32`：层归一化的epsilon。
- `[llm].attention.layer_norm_rms_epsilon: float32`：层RMS归一化的epsilon。

#### RoPE

- `[llm].rope.dimension_count: uint64`：RoPE的旋转维度数量。
- `[llm].rope.freq_base: float32`：RoPE的基础频率。

##### 缩放

以下键描述了RoPE的缩放参数：

- `[llm].rope.scaling.type: string`：可以是 `none`、`linear` 或 `yarn`。
- `[llm].rope.scaling.factor: float32`：用于调整上下文长度的RoPE的缩放因子。
- `[llm].rope.scaling.original_context_length: uint32_t`：基础模型的原始上下文长度。
- `[llm].rope.scaling.finetuned: bool`：如果模型已使用RoPE缩放进行微调，则为True。

请注意，旧模型可能没有这些键，而是可能使用以下键：

- `[llm].rope.scale_linear: float32`：用于调整上下文长度的RoPE的线性缩放因子。

如果可能的话，建议模型使用较新的键，因为它们更灵活，并允许更复杂的缩放方案。执行器需要持续支持这两种键。

#### 模型

以下部分描述了每个模型架构的元数据。每个指定的键 *必须* 存在。

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### 可选项

- `llama.rope.scale`

- `llama.attention.head_count_kv`

- `llama.tensor_data_layout:`

  - `Meta AI original pth:`

    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```

- `llama.expert_count`

- `llama.expert_used_count`


##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### 可选项

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### 可选项

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### 可选项

- `falcon.tensor_data_layout`：

  - `jploski`（Falcon的GGML实现的原作者）：

    ```python
    # 原始的查询键值张量包含n_head_kv "kv组"，
    # 每个组由n_head/n_head_kv个查询权重组成，后面跟着一个键和一个值的权重（由kv组中的所有查询头共享）。
    # 这种布局在GGML中很难处理。
    # 因此，我们在这里重新排列它们，这样我们就有了n_head个查询权重，
    # 然后是n_head_kv个键权重，然后是n_head_kv个值权重，
    # 连续排列。
    
    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
    
        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
    
        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

    

##### RWKV

词汇表大小与 `head` 矩阵中的行数相同。

- `rwkv.architecture_version: uint32`：目前允许的唯一值是4。预计将来会出现版本5。
- `rwkv.context_length: uint64`：在训练或微调过程中使用的上下文长度。RWKV能够处理比此限制更大的上下文，但输出质量可能会受到影响。

### Whisper

未定义类型的键值对应 `llm.` 键的定义。例如，`whisper.context_length` 等同于 `llm.context_length`。这是因为它们都是变压器模型。

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`
- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
- `whisper.decoder.attention.head_count`

#### Prompting

**TODO**: 包括提示格式，和/或有关如何使用它的元数据（指令、对话、自动补全等）。

### LoRA

**TODO**: 确定LoRA所需的元数据。可能希望的特性包括：

- 精确匹配现有模型，以防止错误应用
- 将其标记为LoRA，以防止执行器试图单独运行它

这应该是一个架构，还是应该与原始模型的详细信息共享，并附加字段以将其标记为LoRA？

### Tokenizer

以下键用于描述模型的分词器。建议模型作者尽可能支持这些键，因为它将允许支持的执行器具有更好的分词质量。

#### GGML

GGML支持嵌入式词汇表，可用于推断模型，但是使用此词汇表进行分词的实现（例如 `llama.cpp` 的分词器）可能比模型使用的原始分词器精度更低。当存在并且支持更准确的分词器时，应使用它。

这不保证在模型间标准化，并且可能会在将来更改。建议模型作者尽可能使用更标准化的分词器。

- `tokenizer.ggml.model: string`
  : 分词器模型的名称。
  - `llama`: Llama 风格的 SentencePiece（从 HF `tokenizer.model` 提取的标记和分数）
  - `replit`: Replit 风格的 SentencePiece（从 HF `spiece.model` 提取的标记和分数）
  - `gpt2`: GPT-2 / GPT-NeoX 风格的 BPE（从 HF `tokenizer.json` 提取的标记）
  - `rwkv`: RWKV 分词器

- `tokenizer.ggml.tokens: array[string]`
  : 由模型使用的标记的列表，按照模型使用的标记ID进行索引。

- `tokenizer.ggml.scores: array[float32]`
  : 如果存在，则每个标记的分数/概率。如果不存在，则假定所有标记具有相等的概率。如果存在，则它必须与 `tokens` 具有相同的长度和索引。

- `tokenizer.ggml.token_type: array[int32]`
  : 标记类型（1=正常，2=未知，3=控制，4=用户定义，5=未使用，6=字节）。如果存在，则它必须与 `tokens` 具有相同的长度和索引。

- `tokenizer.ggml.merges: array[string]`
  : 如果存在，则为分词器的合并。如果不存在，则假定标记是原子的。

##### 特殊标记

- `tokenizer.ggml.bos_token_id: uint32`
  : 序列开始标记
- `tokenizer.ggml.eos_token_id: uint32`
  : 序列结束标记
- `tokenizer.ggml.unknown_token_id: uint32`
  : 未知标记
- `tokenizer.ggml.separator_token_id: uint32`
  : 分隔符标记
- `tokenizer.ggml.padding_token_id: uint32`
  : 填充标记

#### Hugging Face

Hugging Face维护着自己的 `tokenizers` 库，支持多种分词器。如果您的执行器使用此库，可能可以直接使用模型的分词器。

- `tokenizer.huggingface.json: string`
  : 给定模型的完整 HF `tokenizer.json`（例如 https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json）。包含是为了与直接支持HF分词器的执行器兼容。

#### 其他

其他分词器可能会被使用，但不一定是标准化的。它们可能是特定于执行器的。随着它们被发现/进一步开发，它们将在此进行记录。

- `tokenizer.rwkv.world: string`
  : 一个RWKV World分词器，类似于[此处](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt)。此文本文件应按原样包含。 
- `tokenizer.chat_template : string`
  : 一个指定模型所期望输入格式的Jinja模板。更多详情参见：https://huggingface.co/docs/transformers/main/en/chat_templating


### 计算图

这部分描述了计算图格式的未来可能的扩展，目前需要讨论，并且可能需要一个新的 GGUF 版本。在写作时，主要的障碍是计算图格式的稳定性。

GGML 节点的样本计算图可以包含在模型本身中，允许执行器运行模型而无需提供其自己的架构实现。这将允许跨执行器实现更一致的体验，并且可以支持更复杂的架构，而无需执行器实现它们。

## 标准张量命名

为了最小化复杂性并最大化兼容性，建议使用转换器架构的模型使用以下命名约定：

### 基础层

```
AA.weight` `AA.bias
```

其中 `AA` 可以是：

- `token_embd`: 令牌嵌入层
- `pos_embd`: 位置嵌入层
- `output_norm`: 输出归一化层
- `output`: 输出层

### 注意力和前馈层块

```
blk.N.BB.weight` `blk.N.BB.bias
```

其中 N 表示层所属的块编号，`BB` 可以是：

- `attn_norm`: 注意力归一化层
- `attn_norm_2`: 注意力归一化层
- `attn_qkv`: 注意力查询-键-值层
- `attn_q`: 注意力查询层
- `attn_k`: 注意力键层
- `attn_v`: 注意力值层
- `attn_output`: 注意力输出层
- `ffn_norm`: 前馈网络归一化层
- `ffn_up`: 前馈网络 "上" 层
- `ffn_gate`: 前馈网络 "门" 层
- `ffn_down`: 前馈网络 "下" 层
- `ffn_gate_inp`: MoE 模型中前馈网络的专家路由层
- `ffn_gate_exp`: MoE 模型中每个专家的前馈网络 "门" 层
- `ffn_down_exp`: MoE 模型中每个专家的前馈网络 "下" 层
- `ffn_up_exp`: MoE 模型中每个专家的前馈网络 "上" 层

## 版本历史

该文档会持续更新以描述元数据的当前状态，这些更改并不会在提交之外进行跟踪。

但是，格式本身已经发生了变化。以下部分描述了格式本身的变化。

### v3

增加了大端支持。

### v2

大多数可计数值（长度等）从 `uint32` 更改为 `uint64`，以便将来支持更大的模型。

### v1

初始版本。

## 历史情况

以下信息提供了一些背景，但并非理解本文档其余内容所必需。

### 概述

目前，有三种 GGML 文件格式用于 LLM（大型语言模型）：

- **GGML**（无版本）：基线格式，没有版本控制或对齐。
- **GGMF**（版本化）：与 GGML 相同，但具有版本控制。只有一个版本存在。
- **GGJT**：对张量进行对齐，以便用于 `mmap`，这需要对齐。v1、v2 和 v3 是相同的，但后续版本使用了与之前版本不兼容的不同量化方案。

GGML 主要由 `ggml` 中的示例使用，而 `llama.cpp` 模型使用 GGJT。其他执行器可能会使用其中任何一种格式，但这并不是"官方"支持的。

这些格式共享相同的基本结构：

- 带有可选版本号的魔数
- 特定于模型的超参数，包括
  - 有关模型的元数据，如层数、头数等
  - 一个描述大多数张量类型的 `ftype`
    - 对于 GGML 文件，量化版本被编码在 `ftype` 中并除以 1000
- 嵌入式词汇表，是一个带有长度前缀的字符串列表。GGMF/GGJT 格式在字符串旁边嵌入了一个 float32 分数。
- 最后是张量列表，包括它们的长度前缀名称、类型和（在 GGJT 的情况下）对齐的张量数据

值得注意的是，这种结构不能标识模型属于哪种模型架构，也不能灵活改变超参数的结构。这意味着添加或删除任何新的超参数都会破坏现有模型。

### 缺陷

不幸的是，在过去的几个月里，现有模型出现了一些问题：

- 无法识别给定模型属于哪种模型架构，因为没有该信息
  - 同样地，现有程序在遇到新架构时无法智能失败
- 添加或删除任何新超参数都会破坏现有模型，读者无法检测到，除非使用启发式方法
- 每种模型架构都需要自己的 GGML

 转换脚本

- 在不破坏格式结构的情况下保持向后兼容性需要巧妙的技巧，例如将量化版本打包到 ftype 中，这并不能保证读者/编写者会注意到，并且在两种格式之间不一致

### 为什么不使用其他格式？

还有一些其他格式可以使用，但存在的问题包括：

- 需要额外的依赖项来加载或保存模型，在 C 环境中这很复杂
- 对 4 位量化的支持有限或没有支持
- 存在文化期望（例如模型是目录还是文件）
- 不支持嵌入式词汇表
- 对未来发展方向缺乏控制

最终，GGUF 可能会在可预见的未来仍然是必需的，最好有一个单一格式，该格式有着良好的文档记录并且得到所有执行器的支持，而不是扭曲现有格式以满足 GGML 的需求。