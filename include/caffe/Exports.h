#ifndef _C_EXPORTS_H_
#define _C_EXPORTS_H_


#ifdef WIN32
#ifdef CINTERFANCE_EXPORTS
#define C_API __declspec(dllexport) 
#else
#define C_API __declspec(dllimport) 
#endif
#else  
#define C_API
#endif


#endif