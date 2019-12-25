#include "CIniReader.h"

#include <iostream>
#include <fstream>
#include <sstream>

/***************************************************************************
*
*	Class Name: CIniReader
*
****************************************************************************/

////////////////////////////////////////////////////////////////////////////
//	Public implementation
////////////////////////////////////////////////////////////////////////////

/***************************************************************************
*
*	Function Name: getReference
*
****************************************************************************/

CIniReader& CIniReader::getReference()
{
	static CIniReader m_CIniReader;
	return m_CIniReader;
}

/***************************************************************************
*
*	Function Name: readLine
*
****************************************************************************/
EReadERR CIniReader::readLine(std::string fileName, int lineNumber, std::string &currLine)
{
	EReadERR rv(EER_SUCCESS);
	std::ifstream iniFile;
	int linePos(0);

	iniFile.open(fileName, std::ios::in);
	if (iniFile.is_open())
	{
		while (1)
		{
			if (getline(iniFile, currLine))
			{
				linePos++;
				if (linePos > lineNumber)
				{
					break;
				}
			}
			else
			{
				rv = EER_EOF;
				break;
			}
		}
	}
	else
	{
		rv = EER_UNDEFINED_FILE;
	}
	return rv;
}

/***************************************************************************
*
*	Function Name: parseLine
*
****************************************************************************/
EReadERR CIniReader::parseLine(const std::string &inputLine, std::string &retName, std::string &retString, float &retFloat, bool &retBool, cv::Rect2f &rect, EReadERR &retRead, bool &isHeader)
{
	EReadERR rv(EER_SUCCESS);

	std::string importantPart;
	std::vector<std::size_t> pos;
	retName = "";
	retString = "";
	retFloat = 0.0;
	retBool = false;
	rect = cv::Rect();
	retRead = EER_SUCCESS;
	isHeader = false;

	if (inputLine.size() != 0)
	{
		//find important part (i.e. without comment)
		rv = findStrImportantPart(inputLine, importantPart);
		if (EER_SUCCESS == rv)
		{
			// check if line is header or data line. Header is marked by [header].
			isHeader = isHeaderLine(importantPart, retName);
			if (!isHeader)
			{
				// parse variable name and value
				parseStringToVals(importantPart, retName, retString, retFloat, retBool, rect, retRead);
			}
		}
	}
	else
	{
		rv = EER_EMPTY_LINE;
	}


	return rv;
}

////////////////////////////////////////////////////////////////////////////
//	Private implementation
////////////////////////////////////////////////////////////////////////////

/***************************************************************************
*
*	Function Name: findSubstringOccurences
*
****************************************************************************/
void CIniReader::findSubstringOccurences(const std::string &inputString, const std::string &subString, std::vector<std::size_t> &positions)
{
	size_t pos = inputString.find(subString, 0);
	while (pos != std::string::npos)
	{
		positions.push_back(pos);
		pos = inputString.find(subString, pos + 1);
	}
}

/***************************************************************************
*
*	Function Name: findStrImportantPart
*
****************************************************************************/
EReadERR CIniReader::findStrImportantPart(const std::string &inputString, std::string &retString)
{
	EReadERR rv(EER_SUCCESS);

	std::vector<std::size_t> occurences;
	findSubstringOccurences(inputString, std::string("#"), occurences);
	if (occurences.size() > 0)
	{
		if (occurences[0] != 0)
		{
			retString.assign(inputString, 0, occurences[0] - 1);
		}
		else
		{
			rv = EER_COMMENT;
		}
	}
	else
	{
		retString = inputString;
	}

	return rv;
}

/***************************************************************************
*
*	Function Name: isHeaderLine
*
****************************************************************************/
bool CIniReader::isHeaderLine(const std::string &inputString, std::string &retString)
{
	bool isHeader = false;
	std::vector<std::size_t> pos, pos1;

	findSubstringOccurences(inputString, std::string("["), pos);
	findSubstringOccurences(inputString, std::string("]"), pos1);

	if (pos.size() != 0 && pos1.size() != 0 ) //if both [] found in string
	{
		if ((pos[0] < pos1[0]) && (pos[0] == 0)) // if the first "[" is in the beginning
		{
			isHeader = true;
			retString.assign(inputString, pos[0]+1, pos1[0] - 1);
		}
	}

	return isHeader;
}

/***************************************************************************
*
*	Function Name: parseStringToVals
*
****************************************************************************/
EReadERR CIniReader::parseStringToVals(const std::string &inputString, std::string &retName, std::string &retString, float &retFloat, bool &retBool, cv::Rect2f &rect, EReadERR &retRead)
{
	EReadERR rv(EER_SUCCESS), rv1(EER_SUCCESS);
	std::vector<std::size_t> pos, spacePos;
	std::string equationPart;
	retString = "";
	retFloat = 0.0;
	retBool = false;
	retRead = EER_SUCCESS;

	findSubstringOccurences(inputString, std::string("="), pos);
	findSubstringOccurences(inputString, std::string(" "), spacePos);
	if (spacePos.size() == 0)
	{
		findSubstringOccurences(inputString, std::string("\t"), spacePos);
	}
	if (pos.size() > 0)
	{
		retName.assign(inputString, 0, pos[0]);
		// convert the other side of equation to number, rectangle or value\string
		if (spacePos.size() == 0)
		{
			equationPart.assign(inputString, pos[0] + 1, inputString.size() - 1);
		}
		else
		{
			equationPart.assign(inputString, pos[0] + 1, (spacePos[0] - pos[0] - 1));
		}
		// call methods
		rv1 = str2rect(equationPart, rect);
		if (EER_INVALID_PARAM == rv1) // if it isn't a rectangle
		{
			rv1 = str2bool(equationPart, retBool);
			if (EER_UNDEFINED_TYPE == rv1) // if it neither rect nor bool is float
			{
				rv1 = str2float(equationPart, retFloat);
				if (EER_SUCCESS != rv1) // if it isn't rect bool or float, it is a string
				{
					retString = equationPart;
					retRead = EER_ELSE;
				}
				else
				{
					retRead = EER_FLOAT;
				}
			}
			else
			{
				retRead = EER_BOOL;
			}
		}
		else
		{
			retRead = EER_RECT;
		}
	}
	else
	{
		rv = EER_UNDEFINED_TYPE;
	}

	return rv;
}


/***************************************************************************
*
*	Function Name: str2float
*
****************************************************************************/
EReadERR CIniReader::str2float(const std::string &inString, float &outVal)
{
	EReadERR rv(EER_SUCCESS);
	const char *convertedInString;
	convertedInString = inString.c_str();
	char *end;
	float  result;
	errno = 0;
	result = strtof(convertedInString, &end);
	if ((errno == ERANGE && result == INFINITY) || result > INFINITY)
	{
		rv = EER_INVALID_DIMS;
	}
	if ((errno == ERANGE && result == -INFINITY) || result < -INFINITY)
	{
		rv = EER_INVALID_DIMS;
	}

	if (*convertedInString == '\0' || *end != '\0')
	{
		rv = EER_INVALID_PARAM;
	}
	if (EER_SUCCESS != rv)
	{
		result = 0;
	}
	outVal = result;
	return rv;
}

/***************************************************************************
*
*	Function Name: str2rect
*
****************************************************************************/
EReadERR CIniReader::str2rect(const std::string &inString, cv::Rect2f &rect)
{
	EReadERR rv(EER_SUCCESS);
	std::vector<std::size_t> pos;
	std::string localStr;

	findSubstringOccurences(inString, std::string("["), pos);
	findSubstringOccurences(inString, std::string("]"), pos);

	if (pos.size() == 2) // if found only once []
	{
		localStr.assign(inString, pos[0] + 1, pos[1] - 1);
		pos.clear();
		findSubstringOccurences(localStr, std::string(","), pos);
		if (pos.size() == 3) // if there's exactly 4 parameters comma delimited
		{
			int startPoint(0), endPoint(0);
			float tmp(0.0);
			for (int i(0); i < pos.size(); i++)
			{
				if (i > 0)
				{
					startPoint = pos[i - 1] + 1;
				}
				endPoint = pos[i] - startPoint;
				rv = str2float(localStr.substr(startPoint, endPoint), tmp);
				if (EER_SUCCESS == rv)
				{
					if (i == 0)
					{
						rect.x = tmp;
					}
					if (i == 1)
					{
						rect.y = tmp;
					}
					if (i == 2)
					{
						rect.width = tmp;
					}
				}
			}
			startPoint = pos[2] + 1;
			endPoint = localStr.size() - startPoint;
			rv = str2float(localStr.substr(startPoint, endPoint), tmp);
			if (EER_SUCCESS == rv)
			{
				rect.height = tmp;
			}
		}
		else
		{
			rv = EER_UNDEFINED_TYPE;
		}
	}
	else
	{
		rv = EER_INVALID_PARAM;
	}

	return rv;
}


/***************************************************************************
*
*	Function Name: str2bool
*
****************************************************************************/
EReadERR CIniReader::str2bool(const std::string &inString, bool &retBool)
{
	EReadERR rv(EER_SUCCESS);

	if (strcmp(inString.c_str(), "true") == 0 || strcmp(inString.c_str(), "True") == 0 ||
		strcmp(inString.c_str(), "false") == 0 || strcmp(inString.c_str(), "false") == 0)
	{
		if (strcmp(inString.c_str(), "true") == 0 || strcmp(inString.c_str(), "True") == 0)
		{
			retBool = true;
		}
		else
		{
			retBool = false;
		}
	}
	else
	{
		retBool = false;
		rv = EER_UNDEFINED_TYPE;
	}

	return rv;
}

CIniReader::CIniReader()
{

}

CIniReader::~CIniReader()
{

}
