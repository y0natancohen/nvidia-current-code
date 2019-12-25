#pragma once

#define CUDA_OPTIM 1

enum EReport
{
	ER_SUCCESS = 0,				// success
	ER_FAIL = 1,				// general failure
	ER_INVALID_INPUT = 2,		// invalid input error
	ER_INVALID_PARAM = 3,		// invalid parameter
	ER_MAX_ITER_EXIT = 4,		// maximal itteration exit - can be warning
	ER_INVALID_DIMS = 5,		// invalid dimentions
	ER_INVALID_TYPE = 6,		// invalid object type
	ER_INVALID_HANDLE = 7,		// invalid handles error
	ER_INVALID_OUTPUT = 8,		// invalid output error
	ER_ZERO_DIVISION = 9,		// zero division error
	ER_INVALID_INIT = 10,		// invalid init
	ER_WARNING_DIMS = 11,		// warning wrong dimentions - using code compensation

	ER_COUNT
};

