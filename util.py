def get_output_size(input_size, kernel_size, stride, padding_size):
    return (input_size - kernel_size + padding_size) // stride + 1

def get_total_padding(input_size, kernel_size, stride, output_size):
    return stride * output_size - input_size - stride + kernel_size

def get_input_size(output_size, kernel_size, stride):
    return (output_size - 1) * stride + kernel_size