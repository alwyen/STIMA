// using standard exceptions
#include <iostream>
#include <exception>

#include <string>

enum Status {
noError, error , noMemoryError , badArgumentError ,
sizeError, nullPtrError , outOfRangeError , dataTypeError ,
stepError , maskSizeError  , anchorError  , singularError,
cpuNotSupportedError , zeroMaskValueError , borderError
};

class Exception: public std::exception
{
  private: std::string msg;

  public:
  Exception  ( const char *  message = "Unknown or unspecified error"  ) {
    msg = message;
  }
  Exception    (    Status     status    ){
    msg = "dummy";
  }

  virtual const char* what() const throw()
  {
    return msg.c_str();
  }

    
};

