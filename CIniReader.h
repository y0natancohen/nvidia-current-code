#pragma once
#ifndef CINI_HEADER_H
#define CINI_HEADER_H

//#include <stdio.h>
//#include <direct.h>
//#include <conio.h>
#include <iostream>
#include <string>
//#include <Windows.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

enum EReadERR
{
	EER_SUCCESS = 0,
	EER_COMMENT = 1,
	EER_UNDEFINED_TYPE = 2,
	EER_UNDEFINED_FILE = 3,
	EER_EOF = 4,
	EER_EMPTY_LINE = 5,
	EER_INVALID_DIMS = 6,
	EER_INVALID_PARAM = 7,
	EER_RECT = 8,
	EER_BOOL = 9,
	EER_FLOAT = 10,
	EER_ELSE = 11,

	EER_COUNT
};

class CIniReader
{
public:
	//methods
	//////////////////////////////////////////////////////////////////////////////
	///	The function getReference is used in order to define the class as singleton
	///		since the constructor and distructor are private.
	/////////////////////////////////////////////////////////////////////////////
	static CIniReader& getReference();

	/****************************************************************************
	*	The fucntion readLine reads a specific line from a given file, and returns
	*		the read string.
	*
	*	INPUT:
	*		fileName - string - full path to file
	*		lineNumber - int - line number to read (start line is 0)
	*		currLine - string - the line read from file
	*
	*	OUTPUT:
	*		EER_UNDEFINED_FILE - if the file could not open for any reason.
	*		EER_EOF - end of file code
	*		EER_SUCCESS - if line read properly
	****************************************************************************/
	EReadERR readLine(std::string fileName, int lineNumber, std::string &currLine);

	/****************************************************************************
	*	The fucntion parseLine parse a string into relevant values regarding the
	*		following .ini file format:
	*			a comment start with "#" and from its location to tnd of line is comment
	*			[section header]
	*			variableName=variableValue #without any spaces between them
	*			variableName=[a,b,c,d] #special case of rectangle
	*
	*	INPUT:
	*		see comments below
	*
	*	OUTPUT:
	*		EER_UNDEFINED_FILE - if the file could not open for any reason.
	*		EER_EOF - end of file code
	*		EER_SUCCESS - if line read properly
	****************************************************************************/
	EReadERR parseLine(const std::string &inputLine, // input string
		std::string &retName,		// returned variable name
		std::string &retString,		// returned variable value if it is string else ""
		float &retFloat,			// returned variable value if it is float else 0.0
		bool &retBool,				// returned variable value if it is bool else false
		cv::Rect2f &rect,			// returned variable value if it is cv::rect else zero rect
		EReadERR &retRead,			// returned read code (see enum above)
		bool &isHeader);			// if retName is the header name it is true, else is false



private:
	//methods
	/****************************************************************************
	*	The fucntion findSubstringOccurences finds occurences of substring within
	*		input string.
	*
	*	INPUT:
	*		inputString - input string to evaluate
	*		subString - substring to search within input string.
	*		positions - vector of locations where iccurences found,
	*			size of positions = 0 if no occurence found.
	*
	*	OUTPUT:
	*
	****************************************************************************/
	void findSubstringOccurences(const std::string &inputString, const std::string &subString, std::vector<std::size_t> &positions);

	/****************************************************************************
	*	The fucntion findStrImportantPart cut comment part from input string and
	*		returns the important part to parse.
	*
	*	INPUT:
	*		inputString - input string to evaluate
	*		retString - important part without comment.
	*
	*
	*	OUTPUT:
	*		EER_COMMENT - if the whole input string is comment
	*		EER_SUCCESS - if returning the important part.
	*
	****************************************************************************/
	EReadERR findStrImportantPart(const std::string &inputString, std::string &retString);

	/****************************************************************************
	*	The fucntion isHeaderLine evaluate if an input string is section header and
	*		returns the header and true\false if the string is header.
	*
	*	INPUT:
	*		inputString - input string to evaluate
	*		retString - header (if it isn't header it returns "").
	*
	*
	*	OUTPUT:
	*		true\false - true if indeed it is a header, false otherwise
	*
	****************************************************************************/
	bool isHeaderLine(const std::string &inputString, std::string &retString);

	/****************************************************************************
	*	The fucntion parseStringToVals parse a string into relevant values regarding the
	*		following .ini file format:
	*			variableName=variableValue #without any spaces between them
	*			variableName=[a,b,c,d] #special case of rectangle
	*
	*	INPUT:
	*		see comments below
	*
	*	OUTPUT:
	*		EER_UNDEFINED_TYPE - if it isn't in recignized format
	*		EER_SUCCESS
	*
	****************************************************************************/
	EReadERR parseStringToVals(const std::string &inputString, //input string which isn't header or comment
		std::string &retName,		//return variable name
		std::string &retString,		//return variable value if it is a string
		float &retFloat,			//return variable value if it is a float
		bool &retBool,				//return variable value if it is a bool
		cv::Rect2f &rect,			//return variable value if it is a cv::rect
		EReadERR &retRead);			//return EReaderERR read code

	/****************************************************************************
	*	The fucntion str2float converts string to float
	*
	*	INPUT:
	*		inString - input string
	*		outVal	 - output value - if error it returns 0 by default
	*
	*	OUTPUT:
	*		EER_INVALID_DIMS - if failed to read or ecceeds limits
	*		EER_INVALID_PARAM - if input is empty string
	*		EER_SUCCESS
	*
	****************************************************************************/
	EReadERR str2float(const std::string &inString, float &outVal);

	/****************************************************************************
	*	The fucntion str2rect converts string to cv::rect (assuming cv::rect format
	*		[x,y,width,height])
	*
	*	INPUT:
	*		inString - input string
	*		rect	 - cv::Rect2f - if error it returns 0 rectangle by default
	*
	*	OUTPUT:
	*		EER_UNDEFINED_TYPE - within [] there are less or more commas then 3
	*		EER_INVALID_PARAM - if input is not recognized format
	*		EER_SUCCESS
	*
	****************************************************************************/
	EReadERR str2rect(const std::string &inString, cv::Rect2f &rect);

	/****************************************************************************
	*	The fucntion str2bool converts string to boolean value true\false
	*
	*	INPUT:
	*		inString - input string
	*		retBool	 - returned boolean value
	*
	*	OUTPUT:
	*		EER_UNDEFINED_TYPE - if value isn't one of the following: True\true\False\false
	*		EER_SUCCESS
	*
	****************************************************************************/
	EReadERR str2bool(const std::string &inString, bool &retBool);

	CIniReader();
	~CIniReader();
};

#endif // !CINI_HEADER_H
