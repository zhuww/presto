#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include "vectors.h"

long long next2_to_n(long long x);
/* Return the first value of 2^n >= x */

int gcd(int a, int b);
/* Return the greatest common divisor of a and b */

char *rmtrail(char *str);
/* Removes trailing space from a string */
 
char *rmlead(char *str);
/* Removes leading space from a string */

char *strlower(char *str);
/* Convert a string to lower case */

char *remove_whitespace(char *str);
/* Remove leading and trailing space from a string */

void split_path_file(char *input, char **path, char **file);
/* This routine splits an input string into a path and */
/* a filename.  Since is allocates the memory for the  */
/* path and filename dynamically, the calling program  */
/* must free both "path" and "file".                   */

int split_root_suffix(char *input, char **root, char **suffix);
/* This routine splits an input string into a root name */
/* + suffix.  Since is allocates the memory for the     */
/* root and suffix dynamically, the calling program     */
/* must free both "root" and "suffix".                  */
/* If the routine finds a suffix, it returns 1, else 0. */

void strtofilename(char *string);
/* Trim spaces off the end of *input and convert */
/* all other spaces into underscores.            */

float *gen_freqs(long numfreqs, double lof, double df);
/* This routine generates a float vector of length numfreqs */
/* with values set to lof, lof+df, lof+2df, ...             */
/* It is normally used when generating a list of freqs      */
/* for an x-y plot of spectral data.                        */

double *gen_dfreqs(long numfreqs, double lof, double df);
/* This routine generates a double vector of length numfreqs */
/* with values set to lof, lof+df, lof+2df, ...              */
/* It is normally used when generating a list of freqs       */
/* for an x-y plot of spectral data.                         */

void i_to_n(int n, double *rl, double *im);
/* Return the real and imaginary portions of i^n */

void rotate_1d(float *data, long numbins, long bins_to_left);
/* Rotates a vector by bins_to_left places to the left.    */
/* numbins is the number of FLOATING points to move.       */

void frotate(float *data, long numbins, float bins_to_left);
/* Rotates a vector by bins_to_left places to the left.    */
/* numbins is the number of FLOATING points to move.       */

void drotate_1d(double *data, long numbins, long bins_to_left);
/* Rotates a vector by bins_to_left places to the left.    */
/* numbins is the number of DOUBLE points to move.         */

void drotate(double *data, long numbins, double bins_to_left);
/* Rotates a vector by bins_to_left places to the left.    */
/* numbins is the number of DOUBLE points to move.         */

void stats(float *x, int n, double *mean, double *var, 
	   double *skew, double *kurt);
/* For a floating point vector, *x, of length n, this routine  */
/* returns the mean, variance, skewness, and kurtosis of *x.   */

void dstats(double *x, int n, double *mean, double *var, 
	    double *skew, double *kurt);
/* For a double precision vector, *x, of length n, this routine  */
/* returns the mean, variance, skewness, and kurtosis of *x.     */

void avg_var(float *x, int n, double *mean, double *var);
/* For a float vector, *x, of length n, this routine  */
/* returns the mean and variance of *x.               */

void davg_dvar(double *x, int n, double *mean, double *var);
/* For a double vector, *x, of length n, this routine  */
/* returns the mean and variance of *x.                */

inline void update_stats(int N, double x, double *min, double *max,
			 double *avg, double *var);
/* Update time series statistics using one-pass technique */

void ra_dec_to_string(char *radec, int h_or_d, int m, double s);
/* Return a properly formatted string containing RA or DEC values   */
/*   radec is a string with J2000 RA  in the format 'hh:mm:ss.ssss' */
/*   or a string with J2000 DEC in the format 'dd:mm:ss.ssss'       */

void ra_dec_from_string(char *radec, int *h_or_d, int *m, double *s);
/* Return a values for hours or degrees, minutes and seconds        */
/* given a properly formatted RA or DEC string.                     */
/*   radec is a string with J2000 RA  in the format 'hh:mm:ss.ssss' */
/*   or a string with J2000 DEC in the format 'dd:mm:ss.ssss'       */

double hms2hours(int hours, int min, double sec);
/* Convert hours, minutes, and seconds of time to hours */

double dms2rad(int deg, int min, double sec);
/* Convert degrees, minutes, and seconds of arc to radians */

double hms2rad(int hour, int min, double sec);
/* Convert hours, minutes, and seconds of arc to radians */

double sphere_ang_diff(double ra1, double dec1, double ra2, double dec2);
/* Returns the angular difference in radians between two sets */
/* of RA and DEC (in radians).                                */

double mjd_sec_diff(int int1, double frac1, int int2, double frac2);
/* Return the difference in seconds between two MJDs (1 - 2) */




