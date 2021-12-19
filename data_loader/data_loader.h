#pragma once

#include <mutex>
#include <sstream>
#include <fstream>

/**
* Calling convention
*/
#ifdef __cplusplus
#   define EXTERNC extern "C"
#else
#   define EXTERNC
#endif
#if defined (_WIN32)
#   define _CDECL __cdecl
#ifdef DLL_EXPORT
#   define DLLExport EXTERNC __declspec(dllexport)
#else
#   define DLLExport EXTERNC __declspec(dllimport)
#endif
#else
#   define _CDECL
#   define DLLExport EXTERNC
#endif

/**
* Lock
*/
#define LOCK std::mutex
#define l_create(x)
#define l_try_lock(x) x.trylock()
#define l_lock(x)     x.lock()
#define l_unlock(x)   x.unlock()

/*
PGN/EPD parallel processor class
*/
#define MAX_FILE_STR   256

class ParallelFile {
public:
    std::stringstream iss;
    LOCK lock;
    unsigned count;
    bool open(const char*);
    ParallelFile() {}
    virtual bool next(char*, bool = false) = 0;
};
class EPD : public ParallelFile {
public:
    bool next(char*, bool = false) override;
};

/**
 * Chess specific
 */
#define RR    0x01
#define LL   -0x01
#define RU    0x11
#define LD   -0x11
#define UU    0x10
#define DD   -0x10
#define LU    0x0f
#define RD   -0x0f


#define RRU   0x12
#define LLD  -0x12
#define LLU   0x0e
#define RRD  -0x0e
#define RUU   0x21
#define LDD  -0x21
#define LUU   0x1f
#define RDD  -0x1f

#define UUU   0x20
#define DDD  -0x20
#define RRR   0x02
#define LLL  -0x02

#define WSC_FLAG            1
#define WLC_FLAG            2
#define BSC_FLAG            4
#define BLC_FLAG            8
#define WSLC_FLAG           3
#define BSLC_FLAG          12
#define WBC_FLAG           15

#define file(x)          ((x) &  7)
#define rank(x)          ((x) >> 4)
#define file64(x)        ((x) &  7)
#define rank64(x)        ((x) >> 3)
#define SQ(x,y)          (((x) << 4) | (y))
#define SQ64(x,y)        (((x) << 3) | (y))
#define SQ32(x,y)        (((x) << 2) | (y))
#define SQ8864(x)        (((x) + ((x) & 7)) >> 1)
#define SQ6488(x)        ((x) + ((x) & 070))
#define SQ6448(x)        ((x) - 8)
#define SQ4864(x)        ((x) + 8)   
#define MIRRORF(sq)      ((sq) ^ 0x07)
#define MIRRORR(sq)      ((sq) ^ 0x70)
#define MIRRORD(sq)      SQ(file(sq),rank(sq))
#define MIRRORF64(sq)    ((sq) ^ 007)
#define MIRRORR64(sq)    ((sq) ^ 070)
#define MIRRORD64(sq)    SQ64(file64(sq),rank64(sq))
#define MV8866(x)        (SQ8864(m_from(x)) | (SQ8864(m_to(x)) << 6))
#define is_light(x)      ((file(x)+rank(x)) & 1)
#define is_light64(x)    ((file64(x)+rank64(x)) & 1)

#define MAX(a, b)        (((a) > (b)) ? (a) : (b))
#define MIN(a, b)        (((a) < (b)) ? (a) : (b))

enum colors {
    white,black,neutral
};
enum chessmen {
    king = 1,queen,rook,bishop,knight,pawn
};
#undef blank
enum occupancy {
    blank,wking,wqueen,wrook,wbishop,wknight,wpawn,
    bking,bqueen,brook,bbishop,bknight,bpawn,elephant
};
enum ranks {
    RANK1,RANK2,RANK3,RANK4,RANK5,RANK6,RANK7,RANK8
};
enum files {
    FILEA,FILEB,FILEC,FILED,FILEE,FILEF,FILEG,FILEH
};
enum egbb_colors {
    _WHITE,_BLACK
};
enum egbb_occupancy {
    _EMPTY,_WKING,_WQUEEN,_WROOK,_WBISHOP,_WKNIGHT,_WPAWN,
    _BKING,_BQUEEN,_BROOK,_BBISHOP,_BKNIGHT,_BPAWN
};