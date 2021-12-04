#include <string.h>
#include <stdio.h>
#include <iostream>
#include <string>

#define DLL_EXPORT
#include "data_loader.h"
#undef DLL_EXPORT


enum {DEFAULT, LCZERO, SIMPLE, QLEARN, NNUE, NONET = -1};
static const int net_channels[] = {32, 112, 12, 32, 16*12+12+2, 0};
static const int nn_type = NNUE;
static const int CHANNELS = net_channels[nn_type];
#define USE_CHW      0
#define USE_SPARSE   1
#define N_K_INDICES 16

/*
   Decode fen
 */
static const char piece_name[] = "_KQRBNPkqrbnp_";
static const char rank_name[] = "12345678";
static const char file_name[] = "abcdefgh";
static const char col_name[] = "WwBb";
static const char cas_name[] = "KQkq";

int decode_fen(const char* fen_str, int* player, int* castle,
        int* fifty, int* move_number, float* oval, int* piece, int* square)
{
    /*decode fen*/
    int sq,index = 2;
    const char* p = fen_str,*pfen;
    for(int r = 7;r >= 0; r--) {
        for(int f = 0;f <= 7;f++) {
            sq = r * 8 + f;
            if((pfen = strchr(piece_name,*p)) != 0) {
                int pc = int(strchr(piece_name,*pfen) - piece_name);
                if(pc == 1) {
                    piece[0] = pc;
                    square[0] = sq;
                } else if(pc == 7) {
                    piece[1] = pc;
                    square[1] = sq;
                } else {
                    piece[index] = pc;
                    square[index] = sq;
                    index++;
                }
            } else if((pfen = strchr(rank_name,*p)) != 0) {
                for(int i = 0;i < pfen - rank_name;i++) {
                    f++;
                }
            } 
            p++;
        }
        p++;
    }
    piece[index] = 0;
    square[index] = 0;

    /*player*/
    if((pfen = strchr(col_name,*p)) != 0)
        *player = ((pfen - col_name) >= 2);
    p++;
    p++;

    /*castling rights*/
    *castle = 0;
    if(*p == '-') {
        p++;
    } else {
        while((pfen = strchr(cas_name,*p)) != 0) {
            *castle |= (1 << (pfen - cas_name));
            p++;
        }
    }
    /*epsquare*/
    int epsquare;
    p++;
    if(*p == '-') {
        epsquare = 0;
        p++;
    } else {
        epsquare = int(strchr(file_name,*p) - file_name);
        p++;
        epsquare += 16 * int(strchr(rank_name,*p) - rank_name);
        p++;
    }
    square[index] = epsquare;

    /*fifty & hply*/
    p++;
    if(*p && *(p+1) && isdigit(*p) && ( isdigit(*(p+1)) || *(p+1) == ' ' ) ) {
        sscanf(p,"%d %d",fifty,move_number);
        int skip = 4 + (*fifty >= 10) + (*move_number >= 10) +
                       (*fifty >= 100) + (*move_number >= 100);
        p += skip;
        if(*move_number <= 0) *move_number = 1;
    } else {
        *fifty = 0;
        *move_number = 1;
    }

    /*result*/
    char res[8];
    sscanf(p,"%s %f",&res[0],oval);
    if(*player > 0)
        *oval = 1 - *oval;

    return index + 1;
}

/*
   Fill input planes
 */
#define invert_color(x)  (((x) > 6) ? ((x) - 6) : ((x) + 6))

void fill_input_planes(
        int player, int cast, int fifty, int hply, int epsquare, bool flip_h, int hist,
        int* const draw, int* const piece, int* const square, float* data, int nn_type_,
        int* const __restrict indices0, int8_t* const __restrict values0,
        int* const __restrict indices1, int8_t* const __restrict values1,
        int* const cidx0, int* const cidx1,
        int pid)
{

    int pc, sq, to;

    /* 
       Add the attack map planes 
     */
#define HWC(sq,C)      (rank(sq) * 8 * CHANNELS + file(sq) * CHANNELS + (C))
#define CHW(sq,C)      ((C) * 64 + SQ8864(sq))
#define IDX(sq,C)      (USE_CHW ? CHW(sq,C) : HWC(sq,C))
#define DHWC(sq,C)     data[HWC(sq,C)]
#define DCHW(sq,C)     data[CHW(sq,C)]
#define D(sq,C)        data[IDX(sq,C)]

#define SD(sq,C,V) if(USE_SPARSE) {     \
    if(V > 0) {                         \
        int midx = IDX(sq,C);           \
        int& vidx = *cidx;              \
        indices[2 * vidx + 0] = pid;    \
        indices[2 * vidx + 1] = midx;   \
        values[vidx] = V;               \
        *cidx = vidx + 1;               \
    }                                   \
} else { D(sq,C) = V; }

#define SET(C,V)  {                         \
    for(int i = 0; i < 64; i++)             \
        data[(C) * 64 + i] = V;             \
}

    if(!USE_SPARSE)
        memset(data,  0, sizeof(float) * 64 * CHANNELS);

    if(nn_type_ == DEFAULT || nn_type_ == QLEARN) {
        uint8_t board[128];
        memset(board, 0, 128);

        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = SQ6488(square[i]);
            if(player == _BLACK) {
                sq = MIRRORR(sq);
                pc = invert_color(pc);
            }
            if(flip_h) {
                sq = MIRRORF(sq);
            }
            piece[i] = pc;
            square[i] = sq;
            board[sq] = pc;
        }

#define NK_MOVES(dir, off) {                \
    to = sq + dir;                          \
    if(!(to & 0x88)) D(to, off) = 1.0f;     \
}

#define BRQ_MOVES(dir, off) {               \
    to = sq + dir;                          \
    while(!(to & 0x88)) {                   \
        D(to, off) = 1.0f;                  \
        if(board[to] != 0) break;           \
        to += dir;                          \
    }                                       \
}

        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = square[i];
            D(sq,pc+11) = 1.0f;
            switch(pc) {
                case wking:
                    NK_MOVES(RU,0);
                    NK_MOVES(LD,0);
                    NK_MOVES(LU,0);
                    NK_MOVES(RD,0);
                    NK_MOVES(UU,0);
                    NK_MOVES(DD,0);
                    NK_MOVES(RR,0);
                    NK_MOVES(LL,0);
                    break;
                case wqueen:
                    BRQ_MOVES(RU,1);
                    BRQ_MOVES(LD,1);
                    BRQ_MOVES(LU,1);
                    BRQ_MOVES(RD,1);
                    BRQ_MOVES(UU,1);
                    BRQ_MOVES(DD,1);
                    BRQ_MOVES(RR,1);
                    BRQ_MOVES(LL,1);
                    break;
                case wrook:
                    BRQ_MOVES(UU,2);
                    BRQ_MOVES(DD,2);
                    BRQ_MOVES(RR,2);
                    BRQ_MOVES(LL,2);
                    break;
                case wbishop:
                    BRQ_MOVES(RU,3);
                    BRQ_MOVES(LD,3);
                    BRQ_MOVES(LU,3);
                    BRQ_MOVES(RD,3);
                    break;
                case wknight:
                    NK_MOVES(RRU,4);
                    NK_MOVES(LLD,4);
                    NK_MOVES(RUU,4);
                    NK_MOVES(LDD,4);
                    NK_MOVES(LLU,4);
                    NK_MOVES(RRD,4);
                    NK_MOVES(RDD,4);
                    NK_MOVES(LUU,4);
                    break;
                case wpawn:
                    NK_MOVES(RU,5);
                    NK_MOVES(LU,5);
                    break;
                case bking:
                    NK_MOVES(RU,6);
                    NK_MOVES(LD,6);
                    NK_MOVES(LU,6);
                    NK_MOVES(RD,6);
                    NK_MOVES(UU,6);
                    NK_MOVES(DD,6);
                    NK_MOVES(RR,6);
                    NK_MOVES(LL,6);
                    break;
                case bqueen:
                    BRQ_MOVES(RU,7);
                    BRQ_MOVES(LD,7);
                    BRQ_MOVES(LU,7);
                    BRQ_MOVES(RD,7);
                    BRQ_MOVES(UU,7);
                    BRQ_MOVES(DD,7);
                    BRQ_MOVES(RR,7);
                    BRQ_MOVES(LL,7);
                    break;
                case brook:
                    BRQ_MOVES(UU,8);
                    BRQ_MOVES(DD,8);
                    BRQ_MOVES(RR,8);
                    BRQ_MOVES(LL,8);
                    break;
                case bbishop:
                    BRQ_MOVES(RU,9);
                    BRQ_MOVES(LD,9);
                    BRQ_MOVES(LU,9);
                    BRQ_MOVES(RD,9);
                    break;
                case bknight:
                    NK_MOVES(RRU,10);
                    NK_MOVES(LLD,10);
                    NK_MOVES(RUU,10);
                    NK_MOVES(LDD,10);
                    NK_MOVES(LLU,10);
                    NK_MOVES(RRD,10);
                    NK_MOVES(RDD,10);
                    NK_MOVES(LUU,10);
                    break;
                case bpawn:
                    NK_MOVES(RD,11);
                    NK_MOVES(LD,11);
                    break;
            }
        }

#undef NK_MOVES
#undef BRQ_MOVES

        /*castling, fifty and on-board mask channels*/
        if(epsquare > 0) {
            sq = epsquare;
            if(player == _BLACK) sq = MIRRORR(sq);
            if(flip_h) sq = MIRRORF(sq);

            D(sq, (CHANNELS - 8)) = 1.0;
        }

        if(player == _BLACK) {
            if(cast & BLC_FLAG) SET((CHANNELS - (flip_h ? 6 : 7) ), 1.0);
            if(cast & BSC_FLAG) SET((CHANNELS - (flip_h ? 7 : 6) ), 1.0);
            if(cast & WLC_FLAG) SET((CHANNELS - (flip_h ? 4 : 5) ), 1.0);
            if(cast & WSC_FLAG) SET((CHANNELS - (flip_h ? 5 : 4) ), 1.0);
        } else {
            if(cast & WLC_FLAG) SET((CHANNELS - (flip_h ? 6 : 7) ), 1.0);
            if(cast & WSC_FLAG) SET((CHANNELS - (flip_h ? 7 : 6) ), 1.0);
            if(cast & BLC_FLAG) SET((CHANNELS - (flip_h ? 4 : 5) ), 1.0);
            if(cast & BSC_FLAG) SET((CHANNELS - (flip_h ? 5 : 4) ), 1.0);
        }
        SET((CHANNELS - 3), (hply / 2 + 1) / 200.0);
        SET((CHANNELS - 2), fifty / 100.0);
        SET((CHANNELS - 1), 1.0);

    } else if (nn_type_ == SIMPLE) {

        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = SQ6488(square[i]);
            if(player == _BLACK) {
                sq = MIRRORR(sq);
                pc = invert_color(pc);
            }
            D(sq,(pc-1)) = 1.0f;
        }

    } else if (nn_type_ == NNUE) {

        static const unsigned KINDEX[] = {
#if N_K_INDICES==32
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31
#elif N_K_INDICES==16
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  8,  9,  9,
            10, 10, 11, 11,
            12, 12, 13, 13,
            12, 12, 13, 13,
            14, 14, 15, 15,
            14, 14, 15, 15
#elif N_K_INDICES==8
            0,  1,  2,  3,
            4,  4,  5,  5,
            6,  6,  6,  6,
            7,  7,  7,  7,
            7,  7,  7,  7,
            7,  7,  7,  7,
            7,  7,  7,  7,
            7,  7,  7,  7
#elif N_K_INDICES==4
            0,  0,  1,  1,
            2,  2,  2,  2,
            3,  3,  3,  3,
            3,  3,  3,  3,
            3,  3,  3,  3,
            3,  3,  3,  3,
            3,  3,  3,  3,
            3,  3,  3,  3
#elif N_K_INDICES==2
                0,  0,  0,  0,
            1,  1,  1,  1,
            1,  1,  1,  1,
            1,  1,  1,  1,
            1,  1,  1,  1,
            1,  1,  1,  1,
            1,  1,  1,  1,
            1,  1,  1,  1
#elif N_K_INDICES==1
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0,
            0,  0,  0,  0
#endif
        };

        //player
        {
            int pl = player;
            bool flip_rank = (pl == _BLACK);

            int ksq = square[pl];
            int f = file64(ksq);
            int r = rank64(ksq);
            bool flip_file = (f < FILEE);
            if(flip_rank) r = RANK8 - r;
            if(flip_file) f = FILEH - f;
            int kindex = KINDEX[r * 4 + (f - FILEE)];

            int* indices = indices0, *cidx = cidx0;
            int8_t *values = values0;
            int rows[6][8] = {0}, cols[6][8] = {0}, ring[6][4] = {0};
            int bishop_dark = 0, knight_dark = 0, pawn_dark = 0, pawn_ring_2_dark = 0;
            int bishop_a1h8 = 0, bishop_a8h1 = 0, bishop_a1h8n = 0, bishop_a8h1n = 0;

            for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
                sq = SQ6488(square[i]);
                if(flip_rank) {
                    sq = MIRRORR(sq);
                    pc = invert_color(pc);
                }
                if(flip_file) {
                    sq = MIRRORF(sq);
                }

                int ix = pc - 1;
                SD(sq,(kindex*12+ix),1);
                SD(sq,(16*12+ix),1);

                if( ix < 6) {
                    f = file(sq);
                    r = rank(sq);

                    rows[ix][f]++;
                    cols[ix][r]++;
                    ring[ix][0]++;
                    if(r >= 1 && r < 7 && f >= 1 && f < 7)
                        ring[ix][1]++;
                    if(r >= 2 && r < 6 && f >= 2 && f < 6)
                        ring[ix][2]++;
                    if(r >= 3 && r < 5 && f >= 3 && f < 5)
                        ring[ix][3]++;

                    if((r + f) % 2 == 0) {
                        if(ix == 3)
                            bishop_dark++;
                        if(ix == 4)
                            knight_dark++;
                        if(ix == 5) {
                            pawn_dark++;
                            if(r >= 2 && r < 6 && f >= 2 && f < 6)
                                pawn_ring_2_dark++;
                        }
                    }

                    if(ix == 3) {
                        if(r == f)
                            bishop_a1h8++; 
                        if(r + f == 7)
                            bishop_a8h1++;  
                        if(r == f + 1 || r + 1 == f)
                            bishop_a1h8n++;  
                        if(r + f == 6 || r + f == 8)
                            bishop_a8h1n++;  
                    }
                }
            }
            for(int ix = 0; ix < 6; ix++) {
                for(int i = 0; i < 8; i++) {
                    SD(SQ(ix, i), (17*12+0), rows[ix][i]);
                    SD(SQ(ix, i), (17*12+1), cols[ix][i]);
                }
                SD(SQ(6, ix), (17*12+0), ring[ix][0]);
                SD(SQ(7, ix), (17*12+0), ring[ix][1]);
                SD(SQ(6, ix), (17*12+1), ring[ix][2]);
                SD(SQ(7, ix), (17*12+1), ring[ix][3]);
            }

            SD(SQ(6, 6), (17*12+0), bishop_dark);
            SD(SQ(6, 7), (17*12+0), knight_dark);
            SD(SQ(7, 6), (17*12+0), pawn_dark);
            SD(SQ(7, 7), (17*12+0), pawn_ring_2_dark);
            SD(SQ(6, 6), (17*12+1), bishop_a1h8);
            SD(SQ(6, 7), (17*12+1), bishop_a8h1);
            SD(SQ(7, 6), (17*12+1), bishop_a1h8n);
            SD(SQ(7, 7), (17*12+1), bishop_a8h1n);
        }

        //shift data array
        float* sdata = data;
        if(!USE_SPARSE) {
            data = data + 64 * CHANNELS;
            memset(data,  0, sizeof(float) * 64 * CHANNELS);
        }
        //opponent
        {
            int pl = 1 - player;
            bool flip_rank = (pl == _BLACK);

            int ksq = square[pl];
            int f = file64(ksq);
            int r = rank64(ksq);
            bool flip_file = (f < FILEE);
            if(flip_rank) r = RANK8 - r;
            if(flip_file) f = FILEH - f;
            int kindex = KINDEX[r * 4 + (f - FILEE)];

            int* indices = indices1, *cidx = cidx1;
            int8_t* values = values1;
            int rows[6][8] = {0}, cols[6][8] = {0}, ring[6][4] = {0};
            int bishop_dark = 0, knight_dark = 0, pawn_dark = 0, pawn_ring_2_dark = 0;
            int bishop_a1h8 = 0, bishop_a8h1 = 0, bishop_a1h8n = 0, bishop_a8h1n = 0;

            for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
                sq = SQ6488(square[i]);
                if(flip_rank) {
                    sq = MIRRORR(sq);
                    pc = invert_color(pc);
                }
                if(flip_file) {
                    sq = MIRRORF(sq);
                }

                int ix = pc - 1;
                SD(sq,(kindex*12+ix),1);
                SD(sq,(16*12+ix),1);

                if( ix < 6) {
                    f = file(sq);
                    r = rank(sq);

                    rows[ix][f]++;
                    cols[ix][r]++;
                    ring[ix][0]++;
                    if(r >= 1 && r < 7 && f >= 1 && f < 7)
                        ring[ix][1]++;
                    if(r >= 2 && r < 6 && f >= 2 && f < 6)
                        ring[ix][2]++;
                    if(r >= 3 && r < 5 && f >= 3 && f < 5)
                        ring[ix][3]++;

                    if((r + f) % 2 == 0) {
                        if(ix == 3)
                            bishop_dark++;
                        if(ix == 4)
                            knight_dark++;
                        if(ix == 5) {
                            pawn_dark++;
                            if(r >= 2 && r < 6 && f >= 2 && f < 6)
                                pawn_ring_2_dark++;
                        }
                    }

                    if(ix == 3) {
                        if(r == f)
                            bishop_a1h8++; 
                        if(r + f == 7)
                            bishop_a8h1++;  
                        if(r == f + 1 || r + 1 == f)
                            bishop_a1h8n++;  
                        if(r + f == 6 || r + f == 8)
                            bishop_a8h1n++;  
                    }
                }
            }
            for(int ix = 0; ix < 6; ix++) {
                for(int i = 0; i < 8; i++) {
                    SD(SQ(ix, i), (17*12+0), rows[ix][i]);
                    SD(SQ(ix, i), (17*12+1), cols[ix][i]);
                }
                SD(SQ(6, ix), (17*12+0), ring[ix][0]);
                SD(SQ(7, ix), (17*12+0), ring[ix][1]);
                SD(SQ(6, ix), (17*12+1), ring[ix][2]);
                SD(SQ(7, ix), (17*12+1), ring[ix][3]);
            }

            SD(SQ(6, 6), (17*12+0), bishop_dark);
            SD(SQ(6, 7), (17*12+0), knight_dark);
            SD(SQ(7, 6), (17*12+0), pawn_dark);
            SD(SQ(7, 7), (17*12+0), pawn_ring_2_dark);
            SD(SQ(6, 6), (17*12+1), bishop_a1h8);
            SD(SQ(6, 7), (17*12+1), bishop_a8h1);
            SD(SQ(7, 6), (17*12+1), bishop_a1h8n);
            SD(SQ(7, 7), (17*12+1), bishop_a8h1n);
        }
        data = sdata;

    } else {

        static const int piece_map[2][12] = {
            {
                wpawn,wknight,wbishop,wrook,wqueen,wking,
                bpawn,bknight,bbishop,brook,bqueen,bking
            },
            {
                bpawn,bknight,bbishop,brook,bqueen,bking,
                wpawn,wknight,wbishop,wrook,wqueen,wking
            }
        };

        for(int h = 0, i = 0; h < hist; h++) {
            for(; (pc = piece[i]) != _EMPTY; i++) {
                sq = SQ6488(square[i]);
                if(player == _BLACK) 
                    sq = MIRRORR(sq);
                int off = piece_map[player][pc - wking]
                    - wking + 13 * h;
                D(sq,off) = 1.0f;
            }
            if(draw && draw[h]) {
                int off = 13 * h + 12;
                SET(off, 1.0);
            }
            i++;
        }

        if(player == _BLACK) {
            if(cast & BLC_FLAG) SET((CHANNELS - 8), 1.0);
            if(cast & BSC_FLAG) SET((CHANNELS - 7), 1.0);
            if(cast & WLC_FLAG) SET((CHANNELS - 6), 1.0);
            if(cast & WSC_FLAG) SET((CHANNELS - 5), 1.0);
            SET((CHANNELS - 4), 1.0);
        } else {
            if(cast & WLC_FLAG) SET((CHANNELS - 8), 1.0);
            if(cast & WSC_FLAG) SET((CHANNELS - 7), 1.0);
            if(cast & BLC_FLAG) SET((CHANNELS - 6), 1.0);
            if(cast & BSC_FLAG) SET((CHANNELS - 5), 1.0);
            SET((CHANNELS - 4), 0.0);
        }
        SET((CHANNELS - 3), fifty / 100.0);
        SET((CHANNELS - 1), 1.0);
    }

#if 0
    for(int c = 0; c < CHANNELS;c++) {
        printf("//Channel %d\n",c);
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < 8; j++) {
                int sq = SQ(i,j);
                printf("%4.2f, ",D(sq,c));
            }
            printf("\n");
        }
        printf("\n");
    }
    fflush(stdout);
#endif
}

/*
 * init planes
 */
static float* inp_planes = 0;

void init_input_planes() {
    unsigned int N_PLANE = (nn_type == NNUE) ? 
        (8 * 8 * net_channels[NNUE] * 2) :
        (8 * 8 * net_channels[LCZERO]);

    inp_planes = (float*) malloc(N_PLANE * sizeof(float));
}

/*
 * fill planes
 */
void fill_data(const char* fen, float* iplanes, int nn_type_,
        int* const __restrict indices0, int8_t* const __restrict values0,
        int* const __restrict indices1, int8_t* const __restrict values1,
        float* const __restrict oval,
        int* const cidx0, int* const cidx1,
        int pid
        ) {
    int pieces[33],squares[33],isdraw[1],player,castle,fifty,move_number, count;
    count = decode_fen((char*)fen,&player,&castle,&fifty,&move_number,oval,pieces,squares);

    bool flip_h = (file64(squares[0]) <= FILED);
    int hist = 1;
    int hply = move_number;
    int epsquare = squares[count - 1];

    ::fill_input_planes(player,castle,fifty,hply,epsquare,flip_h,hist,
            isdraw,pieces,squares,iplanes, nn_type_,
            indices0, values0, indices1, values1, cidx0, cidx1, pid);
}

/*
   Process PGN/EPD in parallel
 */
bool ParallelFile::open(const char* data) {
    iss << data;
    l_create(lock);
    count = 0;
    return true;
}
bool EPD::next(char* moves, bool silent) {
    std::string buffer;
    l_lock(lock);
    if(std::getline(iss,buffer)) {

        strcpy(moves, buffer.c_str());

        count++;
        if(!silent)
            printf("Position %d\t\r",count);
        l_unlock(lock);
        return true;
    }
    l_unlock(lock);
    return false;
}

/*
 * process epd string
 */
DLLExport void _CDECL generate_input_nnue(const char* sdata,
        int* const __restrict indices0, int8_t* const __restrict values0,
        int* const __restrict indices1, int8_t* const __restrict values1,
        float* const __restrict oval,
        int* const cidx0, int* const cidx1
        ) {

    EPD epdf;
    char epds[4 * MAX_FILE_STR];

    epdf.open(sdata);
    *cidx0 = *cidx1 = 0;

    while(epdf.next(epds,true)) {
        int idx = epdf.count - 1;
        fill_data(epds, inp_planes, NNUE,
                indices0, values0,
                indices1, values1,
                &oval[idx],
                cidx0, cidx1,
                idx);
    }
}

/*
   test
 */
#ifdef TEST
int main() {
    static const int N = 131072;

    if(!USE_SPARSE)
        init_input_planes();

    //---read chunk-----
    std::ifstream ifs("temp0.epd");
    std::string all_lines;
    std::string line;
    int cnt = 0;
    while(std::getline(ifs, line)) {
        all_lines += line + "\n";
        cnt++;
        if(cnt >= N) break;
    }

    //---generate input planes
    int* indices0 = new int[N * 192 * 2];
    int* indices1 = new int[N * 192 * 2];
    int8_t* values0 = new int8_t[N * 192];
    int8_t* values1 = new int8_t[N * 192];
    float* oval = new float[N];
    int cidx0, cidx1;

    generate_input_nnue(all_lines.c_str(),
            indices0, values0,
            indices1, values1,
            oval,
            &cidx0, &cidx1);

    return 0;
}
#endif
