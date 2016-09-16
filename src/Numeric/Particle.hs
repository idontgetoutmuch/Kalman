{-# OPTIONS_GHC -Wall                     #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
{-# OPTIONS_GHC -fno-warn-type-defaults   #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind  #-}
{-# OPTIONS_GHC -fno-warn-missing-methods #-}
{-# OPTIONS_GHC -fno-warn-orphans         #-}

{-# LANGUAGE MultiParamTypeClasses        #-}
{-# LANGUAGE TypeFamilies                 #-}
{-# LANGUAGE ScopedTypeVariables          #-}
{-# LANGUAGE ExplicitForAll               #-}
{-# LANGUAGE DataKinds                    #-}


{-# LANGUAGE FlexibleInstances            #-}
{-# LANGUAGE MultiParamTypeClasses        #-}
{-# LANGUAGE FlexibleContexts             #-}
{-# LANGUAGE TypeFamilies                 #-}
{-# LANGUAGE BangPatterns                 #-}
{-# LANGUAGE GeneralizedNewtypeDeriving   #-}
{-# LANGUAGE TemplateHaskell              #-}
{-# LANGUAGE DataKinds                    #-}
{-# LANGUAGE DeriveGeneric                #-}

module Particle where

import           Data.Random hiding ( StdNormal, Normal )
import           Data.Random.Source.PureMT ( pureMT )
import           Control.Monad.State ( evalState, replicateM )
import qualified Control.Monad.Loops as ML
import           Control.Monad.Writer ( tell, WriterT, lift,
                                        runWriterT
                                      )
import           Numeric.LinearAlgebra.Static
                 ( R, vector, Sym,
                   headTail, matrix, sym,
                   diag
                 )
import           GHC.TypeLits ( KnownNat )
import           Data.Random.Distribution.MultivariateNormal ( Normal(..) )
import qualified Data.Vector as V
import           Data.Bits ( shiftR )
import           Data.List ( transpose )
import           Control.Parallel.Strategies
import           GHC.Generics (Generic)

type Particles a = V.Vector a

oneFilteringStep ::
  MonadRandom m =>
  (Particles a -> m (Particles a)) ->
  (Particles a -> Particles b) ->
  (b -> b -> Double) ->
  Particles a ->
  b ->
  WriterT [Particles a] m (Particles a)
oneFilteringStep stateUpdate obsUpdate weight statePrevs obs = do
  tell [statePrevs]
  stateNews <- lift $ stateUpdate statePrevs
  let obsNews = obsUpdate stateNews
  let weights       = V.map (weight obs) obsNews
      cumSumWeights = V.tail $ V.scanl (+) 0 weights
      totWeight     = V.last cumSumWeights
  vs <- lift $ V.replicateM nParticles (sample $ uniform 0.0 totWeight)
  let js = indices cumSumWeights vs
      stateTildes = V.map (stateNews V.!) js
  return stateTildes
