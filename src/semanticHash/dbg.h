#ifndef __dbg_h__
#define __dbg_h__

#include <stdio.h>
#include <errno.h>
#include <string.h>
#define RESET   "\033[0m"
#define RED     "\033[31m"  
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#ifdef NDEBUG
#define debug(M, ...)
#else
#define debug(M, ...) fprintf(stderr,"[" BOLDCYAN "DEBUG" RESET "]" "%s:%d:" M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define clean_errno() (errno == 0 ? "n/a": strerror(errno))

#define log_warn(M, ...) fprintf(stderr, "%-20s[%s:%d] [E: %s] %s"M"\n",\
        "[" BOLDYELLOW "WARN" RESET "]",__FILE__,__LINE__, clean_errno(),"", ##__VA_ARGS__)

#define log_success(M, ...) fprintf(stderr,  "%-20s[%s:%d] %s "M" \n",\
       "[" BOLDGREEN "OK" RESET "]", __FILE__, __LINE__,"", ##__VA_ARGS__)

#define log_info(M, ...) fprintf(stderr,  "%-20s[%s:%d] %s"M "\n",\
       "[" BOLDWHITE "INFO" RESET "]", __FILE__, __LINE__ ,"", ##__VA_ARGS__)

#define log_err(M, ...) fprintf(stderr,  "%-20s[%s:%d] [E: %s] %s"M"\n",\
       "[" BOLDRED "ERROR" RESET "]", __FILE__, __LINE__, clean_errno(),"", ##__VA_ARGS__)
 
#define check(A, M, ...) if(!(A)) { log_err(M, ##__VA_ARGS__); errno=0; goto error; } 
  
#define check_hard(A, M, ...) if(!(A)) { log_err(M, ##__VA_ARGS__); errno=0; exit(1); } 

#define sentinel(M, ...)  { log_err(M, ##__VA_ARGS__); errno=0; goto error; } 
  
#define check_mem(A) check((A), "Out of memory.")
   
#define check_debug(A, M, ...) if(!(A)) { debug(M, ##__VA_ARGS__); errno=0; goto error; } 


#endif
